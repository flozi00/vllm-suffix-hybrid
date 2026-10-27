# SPDX-License-Identifier: Apache-2.0
"""Our own HiSparse residency resolver + written-slot invalidation for SM120
(vLLM 0.31.1 nightly, gate SUFFIX_SM120_HISPARSE_OWN_RESOLVER=1 together with
SUFFIX_SM120_HISPARSE_MTP=1; without it the residency split is used).

Why: vLLM's hisparse_resolve_residency (csrc/libtorch_stable/hisparse_kernels.cu)
resolves all of a request's verify rows in one block whose shared memory holds
the union hash (int32 key + int32 value per entry), the int16 LRU compaction
buffer and the scan scratch: 132,388 B at MTP k=5 (6 rows x top-2048, hot
14,336) where SM120 allows 101,376 B, so stock refuses GLM and our interim fix
runs it in two passes. hisparse_invalidate_written_slots scans a request's whole
hot region with ONE 256-thread block per written token (56 dependent rounds at
hot 14,336).

What: the same kernel, our source, compiled at runtime by NVRTC straight to an
sm_120 cubin (SASS, no PTX: pods never JIT). Two layout changes, no semantic one:
  * union values are 16 bits: a first-reference position (< 0x3FFF) until a
    2-bit tag (hit / miss / invalid) + 14-bit hot slot replaces it; the position
    atomicMin is a CAS on the containing 32-bit word;
  * the int16 LRU compaction buffer lives in global scratch (one row per request
    state, one scratch per index group: groups resolve on their own streams).
  -> 79,140 B at k=5: ONE pass. Phases, orders and tie-breaks are upstream's
  (diff the kernel against hisparse_kernels.cu), so slots, LRU order, swap rows,
  stats and dgi are bit-identical to upstream wherever upstream itself fits.
  Invalidation: one block per (written token, 1024-entry chunk), the same
  per-entry predicate, fully parallel.

    python -m hisparse_mtp_patch.own_resolver   # GPU oracle: exit 0 PASS, 1 FAIL, 3 n/a

Kernel source derived from vLLM's hisparse_kernels.cu (Apache-2.0, Copyright
contributors to the vLLM project), itself a port of SGLang's hisparse kernel.
"""
from __future__ import annotations

import os
import struct
import sys

GATE_ENV = "SUFFIX_SM120_HISPARSE_OWN_RESOLVER"
TAG = "[suffix sm120-hisparse-mtp] own resolver"
POS_LIMIT = 0x3FFF  # 14-bit slots and first-reference positions; 0x3FFF = unset
EDITS = (
    ("_resolve_residency", "        torch.ops._C_cache_ops.hisparse_resolve_residency(\n",
     '        _suffix_hisparse_op(self, "resolve")(\n'),
    ("invalidate_written_slots",
     "        torch.ops._C_cache_ops.hisparse_invalidate_written_slots(\n",
     '        _suffix_hisparse_op(self, "invalidate")(\n'),
)


def gate_enabled() -> bool:
    return os.environ.get(GATE_ENV, "").strip() == "1"


def smem_bytes(union_rows: int, hot_size: int) -> int:
    """Dynamic shared memory: int32 keys, uint16 values (hash_size = union + 1,
    rounded to a word), hit / evict chunk offsets, 5 counters, done bits."""
    hash_size, nbc = union_rows + 1, -(-hot_size // 32)
    return 4 * hash_size + 4 * ((hash_size + 1) // 2) + 4 * (3 * nbc + 7)


def fits(union_rows: int, hot_size: int, available: int) -> bool:
    """Every slot and every position of a run (< buffer chunks * 32) must stay
    below the 14-bit unset value."""
    return (0 < union_rows <= hot_size and -(-hot_size // 32) * 32 < POS_LIMIT
            and smem_bytes(union_rows, hot_size) <= available)


CUDA_SRC = r"""
namespace sfx {
typedef int int32_t;
typedef unsigned int uint32_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

namespace {
constexpr int kWarpSize = 32;
constexpr int32_t kHashEmpty = -1;
// 16-bit union values: a first-reference position (< kUnset) until tag | slot.
constexpr uint16_t kTagMask = 0xC000;
constexpr uint16_t kTagHit = 0x4000;
constexpr uint16_t kTagMiss = 0x8000;
constexpr uint16_t kTagInvalid = 0xC000;
constexpr uint16_t kSlotMask = 0x3FFF;
constexpr uint16_t kUnset = 0x3FFF;
constexpr int kResidencyCounters = 5;
constexpr int kHashLoadInverse = 4;

__device__ __forceinline__ int32_t hash_slot(int32_t key, int size) {
  return static_cast<int32_t>(
      __umulhi(static_cast<uint32_t>(key) * 2654435761u, size));
}

__device__ __forceinline__ int next_slot(int h, int size) {
  return h + 1 == size ? 0 : h + 1;
}

__device__ __forceinline__ int warp_inclusive_scan(int32_t* s_data, int lane_id,
                                                   int offset, int count,
                                                   int accumulator) {
  int idx = lane_id + offset;
  int val = (idx < count) ? s_data[idx] : 0;
#pragma unroll
  for (int i = 1; i < 32; i *= 2) {
    int n = __shfl_up_sync(0xffffffff, val, i);
    if (lane_id >= i) val += n;
  }
  val += accumulator;
  if (idx < count) {
    s_data[idx] = val;
  }
  return __shfl_sync(0xffffffff, val, 31);
}

__device__ __forceinline__ int64_t
get_physical_hot_row(const int32_t* hot_block_table, int32_t row,
                     int64_t table_stride, int32_t block_size, int32_t slot) {
  const int32_t block =
      hot_block_table[static_cast<int64_t>(row) * table_stride +
                      slot / block_size];
  return static_cast<int64_t>(block) * block_size + slot % block_size;
}

__device__ __forceinline__ void store_hot_index(
    int32_t* hot_indices, int32_t* attention_indices, int32_t index,
    int32_t physical_row, int32_t hot_block_size,
    int64_t attention_block_stride) {
  hot_indices[index] = physical_row;
  if (attention_indices != nullptr) {
    attention_indices[index] =
        physical_row < 0
            ? -1
            : (physical_row / hot_block_size) * attention_block_stride +
                  physical_row % hot_block_size;
  }
}

__device__ __forceinline__ int32_t translate_topk_entry(
    const int32_t token_index, const int32_t request_row,
    const int32_t state_row, const int32_t* __restrict__ source_block_table,
    const int32_t* __restrict__ resident_block_table, const int64_t host_rows,
    const int64_t source_bt_stride, const int32_t source_num_reqs,
    const int32_t source_num_blocks, const int32_t source_block_size,
    const int64_t resident_bt_stride, const int32_t resident_num_reqs,
    const int32_t resident_num_blocks, const int32_t resident_block_size,
    const int32_t resident_null_block, int32_t& resident_row) {
  int32_t g = token_index;
  resident_row = -1;
  if (source_block_table != nullptr) {
    const int32_t source_block =
        token_index >= 0 ? token_index / source_block_size : -1;
    if (request_row >= 0 && request_row < source_num_reqs &&
        source_block >= 0 && source_block < source_num_blocks) {
      const int32_t physical_block =
          source_block_table[static_cast<int64_t>(request_row) *
                                 source_bt_stride +
                             source_block];
      g = physical_block > 0 ? physical_block * source_block_size +
                                   token_index % source_block_size
                             : -1;
    } else {
      g = -1;
    }
    if (resident_block_table != nullptr) {
      const int32_t resident_block =
          token_index >= 0 ? token_index / resident_block_size : -1;
      if (state_row >= 0 && state_row < resident_num_reqs &&
          resident_block >= 0 && resident_block < resident_num_blocks) {
        const int32_t physical_block =
            resident_block_table[static_cast<int64_t>(state_row) *
                                     resident_bt_stride +
                                 resident_block];
        if (physical_block != resident_null_block && physical_block >= 0) {
          resident_row = physical_block * resident_block_size +
                         token_index % resident_block_size;
        }
      }
    }
  }
  return g >= host_rows ? -1 : g;
}

// 16-bit atomicMin on shared memory: CAS on the containing 32-bit word.
__device__ __forceinline__ void atomic_min_u16(uint16_t* base, int idx,
                                               uint16_t v) {
  uint32_t* word = reinterpret_cast<uint32_t*>(base) + (idx >> 1);
  const int shift = (idx & 1) * 16;
  uint32_t old = *word, assumed;
  do {
    assumed = old;
    if (((assumed >> shift) & 0xFFFFu) <= v) return;
    const uint32_t next = (assumed & ~(0xFFFFu << shift)) |
                          (static_cast<uint32_t>(v) << shift);
    old = atomicCAS(word, assumed, next);
  } while (old != assumed);
}

__device__ __forceinline__ void write_entry_index(
    const uint16_t entry, const int row, const int i, const int32_t top_k,
    int32_t* __restrict__ hot_indices, int32_t* __restrict__ attention_indices,
    const int64_t attention_row_stride, int32_t* __restrict__ miss_mask,
    const int32_t* __restrict__ hot_block_table, const int request_row,
    const int64_t hot_table_stride, const int32_t hot_block_size,
    const int64_t attention_block_stride, const bool first_reference, int& hits,
    int& misses) {
  int32_t* row_out = hot_indices + static_cast<int64_t>(row) * top_k;
  int32_t* row_attention =
      attention_indices != nullptr
          ? attention_indices + static_cast<int64_t>(row) * attention_row_stride
          : nullptr;
  if ((entry & kTagMask) == kTagInvalid) {
    store_hot_index(row_out, row_attention, i, -1, hot_block_size,
                    attention_block_stride);
    return;
  }
  store_hot_index(row_out, row_attention, i,
                  static_cast<int32_t>(get_physical_hot_row(
                      hot_block_table, request_row, hot_table_stride,
                      hot_block_size, entry & kSlotMask)),
                  hot_block_size, attention_block_stride);
  if ((entry & kTagMask) == kTagMiss && first_reference) {
    if (miss_mask != nullptr) {
      miss_mask[static_cast<int64_t>(row) * top_k + i] = 1;
    }
    ++misses;
  } else {
    ++hits;
  }
}
}  // namespace

// One block per launch row; the first row of each request's contiguous run
// resolves the union of the run's host rows (upstream phases 1-5).
extern "C" __global__ __launch_bounds__(1024) void suffix_hisparse_resolve(
    const int32_t* __restrict__ hot_block_table,
    const int32_t* __restrict__ global_indices,
    const int32_t* __restrict__ request_ids,
    const int32_t* __restrict__ source_block_table,
    const int32_t* __restrict__ resident_block_table,
    int32_t* __restrict__ resolved_global_indices,
    int32_t* __restrict__ valid_counts,
    int32_t* __restrict__ swap_host_physical_rows,
    int32_t* __restrict__ swap_device_physical_rows,
    int32_t* __restrict__ swap_counts,
    int32_t* __restrict__ hot_indices,
    int32_t* __restrict__ attention_indices,
    int32_t* __restrict__ miss_mask,
    int32_t* __restrict__ device_global_indices,
    int16_t* __restrict__ lru_slots,
    int16_t* __restrict__ lru_scratch,
    uint64_t* __restrict__ stats,
    const int32_t* __restrict__ request_state_indices,
    const int32_t request_state_count, const int64_t host_rows,
    const int64_t hot_table_stride, const int32_t hot_block_size,
    const int32_t top_k, const int32_t hot_size, const int32_t hash_size,
    const int64_t region_stride, const int64_t attention_block_stride,
    const int64_t source_bt_stride, const int32_t source_num_reqs,
    const int32_t source_num_blocks, const int32_t source_block_size,
    const int64_t resident_bt_stride, const int32_t resident_num_reqs,
    const int32_t resident_num_blocks, const int32_t resident_block_size,
    const int32_t resident_null_block, const int64_t input_row_stride,
    const int64_t attention_row_stride, const int64_t valid_count_stride) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int num_buffer_chunks = (hot_size + kWarpSize - 1) / kWarpSize;

  const int first_row = blockIdx.x;
  const int request_row =
      request_ids != nullptr ? request_ids[first_row] : first_row;
  if (request_ids != nullptr && first_row > 0 &&
      request_ids[first_row - 1] == request_row) {
    return;
  }
  int end_row = first_row + 1;
  if (request_ids != nullptr) {
    while (end_row < static_cast<int>(gridDim.x) &&
           request_ids[end_row] == request_row) {
      ++end_row;
    }
  }
  const int state_row =
      request_state_indices != nullptr && request_row >= 0 &&
              request_row < request_state_count
          ? request_state_indices[request_row]
          : (request_state_indices == nullptr ? request_row : -1);
  if (state_row < 0) {
    for (int row = first_row; row < end_row; ++row) {
      for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        const int64_t index = static_cast<int64_t>(row) * top_k + i;
        hot_indices[index] = -1;
        if (attention_indices != nullptr) {
          attention_indices[static_cast<int64_t>(row) * attention_row_stride +
                            i] = -1;
        }
        if (resolved_global_indices != nullptr) {
          resolved_global_indices[index] = -1;
        }
      }
      if (valid_counts != nullptr && threadIdx.x == 0) {
        valid_counts[static_cast<int64_t>(row) * valid_count_stride] = 0;
      }
      if (swap_counts != nullptr && threadIdx.x == 0) {
        swap_counts[row] = 0;
      }
    }
    return;
  }
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const unsigned int lanes_before = ((unsigned int)1 << lane_id) - 1;

  int32_t* row_dgi =
      device_global_indices + static_cast<int64_t>(state_row) * region_stride;
  int16_t* row_lru = lru_slots + static_cast<int64_t>(state_row) * hot_size;
  int16_t* s_lru_out = lru_scratch + static_cast<int64_t>(state_row) * hot_size;

  const int table_size =
      min(hash_size, kHashLoadInverse * (end_row - first_row) * top_k);

  extern __shared__ char smem_raw[];
  int32_t* s_hash_keys = reinterpret_cast<int32_t*>(smem_raw);
  uint16_t* s_hash_vals = reinterpret_cast<uint16_t*>(s_hash_keys + hash_size);
  int32_t* s_chunk_off =
      reinterpret_cast<int32_t*>(s_hash_vals) + (hash_size + 1) / 2;
  int32_t* s_evict_off = s_chunk_off + (num_buffer_chunks + 1);
  int32_t* s_counters = s_evict_off + (num_buffer_chunks + 1);
  unsigned int* s_done =
      reinterpret_cast<unsigned int*>(s_counters + kResidencyCounters);

  for (int i = tid; i < table_size; i += blockDim.x) {
    s_hash_keys[i] = kHashEmpty;
    s_hash_vals[i] = kUnset;
  }
  for (int i = tid; i < num_buffer_chunks + 1; i += blockDim.x) {
    s_chunk_off[i] = 0;
    s_evict_off[i] = 0;
  }
  if (tid < kResidencyCounters) {
    s_counters[tid] = 0;
  }
  __syncthreads();

  // Phase 1: translate, resolve resident rows, collect the union of host rows
  // keyed by their first (row, top-k) position.
  for (int row = first_row; row < end_row; ++row) {
    const int32_t* row_topk =
        global_indices + static_cast<int64_t>(row) * input_row_stride;
    int32_t* row_out = hot_indices + static_cast<int64_t>(row) * top_k;
    int32_t* row_attention =
        attention_indices != nullptr
            ? attention_indices +
                  static_cast<int64_t>(row) * attention_row_stride
            : nullptr;
    int32_t* row_miss = miss_mask != nullptr
                            ? miss_mask + static_cast<int64_t>(row) * top_k
                            : nullptr;
    for (int i = tid; i < top_k; i += blockDim.x) {
      int32_t resident_row;
      const int32_t g = translate_topk_entry(
          row_topk[i], request_row, state_row, source_block_table,
          resident_block_table, host_rows, source_bt_stride, source_num_reqs,
          source_num_blocks, source_block_size, resident_bt_stride,
          resident_num_reqs, resident_num_blocks, resident_block_size,
          resident_null_block, resident_row);
      if (resolved_global_indices != nullptr) {
        resolved_global_indices[static_cast<int64_t>(row) * top_k + i] = g;
      }
      if (resident_row >= 0 || g >= 0) atomicAdd(&s_counters[2], 1);
      if (row_miss != nullptr) row_miss[i] = 0;
      if (resident_row >= 0) {
        store_hot_index(row_out, row_attention, i, resident_row, hot_block_size,
                        attention_block_stride);
      } else if (g < 0) {
        store_hot_index(row_out, row_attention, i, -1, hot_block_size,
                        attention_block_stride);
      } else {
        const int position = (row - first_row) * top_k + i;
        int h = hash_slot(g, table_size);
        bool inserted = false;
        for (int probe = 0; probe < table_size; ++probe) {
          const int32_t old = atomicCAS(&s_hash_keys[h], kHashEmpty, g);
          if (old == kHashEmpty) {
            atomicAdd(&s_counters[1], 1);
            inserted = true;
            break;
          }
          if (old == g) {
            inserted = true;
            break;
          }
          h = next_slot(h, table_size);
        }
        if (inserted && position < kUnset) {
          atomic_min_u16(s_hash_vals, h, static_cast<uint16_t>(position));
        } else {
          // Table full, or a position past the 14-bit encoding: trap below.
          atomicAdd(&s_counters[1], hash_size);
        }
      }
    }
    __syncthreads();
    if (tid == 0) {
      if (valid_counts != nullptr) {
        valid_counts[static_cast<int64_t>(row) * valid_count_stride] =
            s_counters[2];
      }
      s_counters[2] = 0;
    }
    __syncthreads();
  }
  if (s_counters[1] >= hash_size) {
    __trap();
  }
  if (s_counters[1] == 0) {
    for (int row = first_row + tid; row < end_row; row += blockDim.x) {
      if (swap_counts != nullptr) swap_counts[row] = 0;
    }
    return;
  }
  // Upstream contract: a run with host rows never covers more positions than
  // the buffer chunks (s_chunk_off / s_done are sized by them).
  if ((end_row - first_row) * top_k > num_buffer_chunks * kWarpSize) {
    __trap();
  }

  // Phase 2: walk hot slots in LRU order; hits compact forward, evictables
  // backward into s_lru_out; a hit tags its union entry with its slot.
  const int iters_buffer = (num_buffer_chunks + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  int total_evict_count = 0;
  for (int iter = 0; iter < iters_buffer; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < num_buffer_chunks;

    const int pos = chunk_idx * kWarpSize + lane_id;
    const bool has_valid_pos = has_valid_chunk && (pos < hot_size);
    const int16_t slot = has_valid_pos ? row_lru[pos] : int16_t(-1);
    const int32_t cached_g =
        (slot >= 0 && slot < region_stride) ? row_dgi[slot] : -1;

    bool is_hit = false;
    if (cached_g >= 0) {
      int h = hash_slot(cached_g, table_size);
      while (true) {
        const int32_t k = s_hash_keys[h];
        if (k == cached_g) {
          s_hash_vals[h] = kTagHit | static_cast<uint16_t>(slot);
          is_hit = true;
          break;
        }
        if (k == kHashEmpty) break;
        h = next_slot(h, table_size);
      }
    }
    const bool is_evictable = has_valid_pos && !is_hit;

    int local_hit_off = 0;
    int local_evict_off = 0;
    if (has_valid_chunk) {
      const unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, is_hit);
      const unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
      local_hit_off = __popc(hit_mask & lanes_before);
      local_evict_off = __popc(evict_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_off[chunk_idx + 1] = __popc(hit_mask);
        s_evict_off[chunk_idx + 1] = __popc(evict_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_hit_count =
          warp_inclusive_scan(s_chunk_off, lane_id, chunk_idx + 1,
                              num_buffer_chunks + 1, total_hit_count);
      total_evict_count =
          warp_inclusive_scan(s_evict_off, lane_id, chunk_idx + 1,
                              num_buffer_chunks + 1, total_evict_count);
      if (tid == 0) {
        s_counters[0] = total_hit_count;
      }
    }
    __syncthreads();

    if (is_hit) {
      s_lru_out[s_chunk_off[chunk_idx] + local_hit_off] = slot;
    }
    if (is_evictable) {
      s_lru_out[hot_size - 1 - (s_evict_off[chunk_idx] + local_evict_off)] =
          slot;
    }
  }
  __syncthreads();

  for (int i = tid; i < num_buffer_chunks + 1; i += blockDim.x) {
    s_chunk_off[i] = 0;
  }
  __syncthreads();

  // Phase 3: positions in (row, top-k) order; an entry's first reference that
  // no slot holds is a miss and takes the oldest evictable slot.
  const int num_positions = (end_row - first_row) * top_k;
  const int num_position_chunks = (num_positions + kWarpSize - 1) / kWarpSize;
  const int iters_position = (num_position_chunks + NUM_WARPS - 1) / NUM_WARPS;
  int miss_running_total = 0;
  int entry_hits = 0;
  int entry_misses = 0;
  for (int iter = 0; iter < iters_position; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < num_position_chunks;

    const int position = chunk_idx * kWarpSize + lane_id;
    int32_t g = -1;
    int h = -1;
    uint16_t entry = 0;
    bool is_miss = false;
    if (has_valid_chunk && position < num_positions) {
      const int row = first_row + position / top_k;
      int32_t resident_row;
      g = translate_topk_entry(
          global_indices[static_cast<int64_t>(row) * input_row_stride +
                         position % top_k],
          request_row, state_row, source_block_table, resident_block_table,
          host_rows, source_bt_stride, source_num_reqs, source_num_blocks,
          source_block_size, resident_bt_stride, resident_num_reqs,
          resident_num_blocks, resident_block_size, resident_null_block,
          resident_row);
      if (resident_row < 0 && g >= 0) {
        h = hash_slot(g, table_size);
        while (s_hash_keys[h] != g) {
          h = next_slot(h, table_size);
        }
        entry = s_hash_vals[h];
        is_miss = entry == position;
      }
    }

    int local_miss_off = 0;
    if (has_valid_chunk) {
      const unsigned int miss_mask_bits = __ballot_sync(0xFFFFFFFF, is_miss);
      local_miss_off = __popc(miss_mask_bits & lanes_before);
      if (lane_id == 0) {
        s_chunk_off[chunk_idx + 1] = __popc(miss_mask_bits);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      miss_running_total =
          warp_inclusive_scan(s_chunk_off, lane_id, chunk_idx + 1,
                              num_position_chunks + 1, miss_running_total);
      if (tid == 0) {
        s_counters[3] = miss_running_total;
      }
    }
    __syncthreads();

    if (is_miss) {
      const int m = s_chunk_off[chunk_idx] + local_miss_off;
      const int16_t evict_slot = s_lru_out[hot_size - 1 - m];
      const int64_t compact_index = static_cast<int64_t>(first_row) * top_k + m;
      if (evict_slot < 0 || evict_slot >= region_stride) {
        s_hash_vals[h] = kTagInvalid;
        if (swap_host_physical_rows != nullptr) {
          swap_host_physical_rows[compact_index] = g;
          swap_device_physical_rows[compact_index] = -1;
        }
      } else {
        s_hash_vals[h] = kTagMiss | static_cast<uint16_t>(evict_slot);
        if (swap_host_physical_rows != nullptr) {
          swap_host_physical_rows[compact_index] = g;
          swap_device_physical_rows[compact_index] =
              static_cast<int32_t>(get_physical_hot_row(
                  hot_block_table, request_row, hot_table_stride,
                  hot_block_size, evict_slot));
        }
        row_dgi[evict_slot] = g;
      }
      entry = s_hash_vals[h];
    }

    const bool resolved = h >= 0 && (entry & kTagMask) != 0;
    if (resolved) {
      write_entry_index(
          entry, first_row + position / top_k, position % top_k, top_k,
          hot_indices, attention_indices, attention_row_stride, miss_mask,
          hot_block_table, request_row, hot_table_stride, hot_block_size,
          attention_block_stride, is_miss, entry_hits, entry_misses);
    }
    if (has_valid_chunk) {
      const unsigned int done_bits = __ballot_sync(0xFFFFFFFF, resolved);
      if (lane_id == 0) {
        s_done[chunk_idx] = done_bits;
      }
    }
  }
  __syncthreads();

  const int total_hits = s_counters[0];
  const int total_misses = s_counters[3];
  if (swap_counts != nullptr) {
    for (int row = first_row + tid; row < end_row; row += blockDim.x) {
      const int remaining = total_misses - (row - first_row) * top_k;
      swap_counts[row] = max(0, min(top_k, remaining));
    }
  }

  // Phase 4: references phase 3 left unresolved map onto the union's slots.
  if (end_row - first_row > 1) {
    for (int position = tid; position < num_positions; position += blockDim.x) {
      if (s_done[position / kWarpSize] & (1u << (position % kWarpSize))) {
        continue;
      }
      const int row = first_row + position / top_k;
      int32_t resident_row;
      const int32_t g = translate_topk_entry(
          global_indices[static_cast<int64_t>(row) * input_row_stride +
                         position % top_k],
          request_row, state_row, source_block_table, resident_block_table,
          host_rows, source_bt_stride, source_num_reqs, source_num_blocks,
          source_block_size, resident_bt_stride, resident_num_reqs,
          resident_num_blocks, resident_block_size, resident_null_block,
          resident_row);
      if (resident_row >= 0 || g < 0) continue;
      int h = hash_slot(g, table_size);
      while (s_hash_keys[h] != g) {
        h = next_slot(h, table_size);
      }
      write_entry_index(
          s_hash_vals[h], row, position % top_k, top_k, hot_indices,
          attention_indices, attention_row_stride, miss_mask, hot_block_table,
          request_row, hot_table_stride, hot_block_size, attention_block_stride,
          false, entry_hits, entry_misses);
    }
  }
  if (stats != nullptr) {
    atomicAdd(&s_counters[4], entry_hits);
    atomicAdd(&s_counters[2], entry_misses);
    __syncthreads();
    if (tid == 0) {
      atomicAdd(&stats[0], static_cast<uint64_t>(s_counters[4]));
      atomicAdd(&stats[1], static_cast<uint64_t>(s_counters[2]));
    }
  }

  // Phase 5: stale evictables first, freshly loaded misses next, hits at MRU.
  const int total_evictable = hot_size - total_hits;
  const int remaining_evictable = total_evictable - total_misses;
  for (int i = tid; i < hot_size; i += blockDim.x) {
    if (i < remaining_evictable) {
      row_lru[i] = s_lru_out[hot_size - 1 - total_misses - i];
    } else if (i < remaining_evictable + total_misses) {
      row_lru[i] = s_lru_out[hot_size - 1 - (i - remaining_evictable)];
    } else {
      row_lru[i] = s_lru_out[i - remaining_evictable - total_misses];
    }
  }
}

// Drop hot copies of re-written KV slots: one block per (token, 1024-entry
// chunk of its request's region), upstream's per-entry predicate.
extern "C" __global__ void suffix_hisparse_invalidate(
    int32_t* __restrict__ device_global_indices,
    const int32_t* __restrict__ request_state_indices,
    const int32_t* __restrict__ req_id_per_token,
    const int64_t* __restrict__ written_slots, const int64_t num_tokens,
    const int64_t num_request_ids, const int64_t num_state_rows,
    const int64_t region_stride) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  const int32_t req_idx = req_id_per_token[token_idx];
  if (req_idx < 0 || req_idx >= num_request_ids) return;
  const int32_t state_idx = request_state_indices[req_idx];
  const int64_t written_slot = written_slots[token_idx];
  if (state_idx < 0 || state_idx >= num_state_rows || written_slot < 0) return;
  int32_t* row = device_global_indices + state_idx * region_stride;
  const int64_t start = static_cast<int64_t>(blockIdx.y) * 1024;
  const int64_t stop = start + 1024 < region_stride ? start + 1024 : region_stride;
  for (int64_t offset = start + threadIdx.x; offset < stop; offset += blockDim.x) {
    if (row[offset] == written_slot) row[offset] = -1;
  }
}
}  // namespace sfx
"""

# Kernel parameter buffers (CU_LAUNCH_PARAM_BUFFER_POINTER): C layout with
# natural alignment = struct's native mode. P pointer, i int32, q int64.
_RESOLVE_PARAMS = struct.Struct("@" + "P" * 18 + "iqqiiiiqqqiiiqiiiiqqq")
_INVALIDATE_PARAMS = struct.Struct("@PPPPqqqq")
_KERNELS: dict = {}  # device index -> (libcuda, {name: CUfunction}, module, cubin)


def _nvrtc():
    """libnvrtc.so.<CUDA major> (torch's CUDA wheel preloads it), else any on disk."""
    import ctypes
    import glob

    import torch

    site = os.path.dirname(os.path.dirname(torch.__file__))
    names = [f"libnvrtc.so.{torch.version.cuda.split('.')[0]}", "libnvrtc.so"]
    names += sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib", "libnvrtc.so*")))
    names += sorted(glob.glob("/usr/local/cuda*/lib64/libnvrtc.so*"))
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.nvrtcGetErrorString.restype = ctypes.c_char_p
        return lib
    raise OSError(f"own HiSparse resolver: no NVRTC library (tried {names})")


def compile_cubin(arch: str) -> bytes:
    """CUDA_SRC -> cubin (SASS for `arch`, e.g. sm_120; no PTX in the image)."""
    import ctypes

    nvrtc = _nvrtc()

    def check(rc):
        if rc != 0:
            raise RuntimeError(f"NVRTC error {rc}: {nvrtc.nvrtcGetErrorString(rc)}")

    prog = ctypes.c_void_p()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), CUDA_SRC.encode(), b"suffix_hisparse.cu",
                                   0, None, None))
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17"]
    try:
        if nvrtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)):
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC compile failed:\n{log.value.decode()[-3000:]}")
        size = ctypes.c_size_t()
        check(nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
        cubin = ctypes.create_string_buffer(size.value)
        check(nvrtc.nvrtcGetCUBIN(prog, cubin))
        return cubin.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


def _check(cuda, rc: int, what: str) -> None:
    if rc != 0:
        import ctypes

        msg = ctypes.c_char_p()
        cuda.cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f"own HiSparse resolver: {what}: CUDA error {rc} "
                           f"({(msg.value or b'?').decode()})")


def _kernels(device_index: int):
    """Our cubin loaded on the device's primary context, once per process and
    device. Call outside CUDA-graph capture (the runtime __init__ does)."""
    if device_index in _KERNELS:
        return _KERNELS[device_index]
    import ctypes

    import torch

    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = compile_cubin(f"sm_{major}{minor}")
    cuda = ctypes.CDLL("libcuda.so.1")
    torch.empty(1, device=torch.device("cuda", device_index))  # primary context current
    with torch.cuda.device(device_index):
        module = ctypes.c_void_p()
        _check(cuda, cuda.cuModuleLoadData(ctypes.byref(module), cubin), "cuModuleLoadData")
        fns = {}
        for name in ("suffix_hisparse_resolve", "suffix_hisparse_invalidate"):
            fn = ctypes.c_void_p()
            _check(cuda, cuda.cuModuleGetFunction(ctypes.byref(fn), module, name.encode()), name)
            fns[name] = fn
        optin = torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin
        _check(cuda, cuda.cuFuncSetAttribute(  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            fns["suffix_hisparse_resolve"], 8, optin), "cuFuncSetAttribute")
    _KERNELS[device_index] = (cuda, fns, module, cubin)
    return _KERNELS[device_index]


def _launch(device_index: int, name: str, grid, block: int, smem: int, params: bytes) -> None:
    import ctypes

    import torch

    cuda, fns, _module, _cubin = _kernels(device_index)
    buf = ctypes.create_string_buffer(params, len(params))
    size = ctypes.c_size_t(len(params))
    extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buf), 2, ctypes.addressof(size), 0)
    stream = ctypes.c_void_p(torch.cuda.current_stream(device_index).cuda_stream)
    _check(cuda, cuda.cuLaunchKernel(fns[name], grid[0], grid[1], 1, block, 1, 1, smem, stream,
                                     None, extra), name)


def _ptr(t) -> int:
    return 0 if t is None else t.data_ptr()


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"own HiSparse resolver: {what}")


def resolve_residency(host_cache, hot_cache, hot_block_table, global_indices, hot_indices,
                      device_global_indices, lru_slots, request_state_indices, region_stride,
                      max_union_rows, miss_mask=None, stats=None, attention_indices=None,
                      attention_block_stride=0, request_ids=None, source_block_table=None,
                      source_block_size=0, resolved_global_indices=None, valid_counts=None,
                      swap_host_physical_rows=None, swap_device_physical_rows=None,
                      swap_counts=None, resident_block_table=None, resident_block_size=0,
                      resident_null_block=0, *, lru_scratch) -> None:
    """torch.ops._C_cache_ops.hisparse_resolve_residency, same arguments, plus the
    index group's LRU compaction scratch (lru_slots' shape and dtype)."""
    import torch

    rows, top_k = global_indices.shape
    hot_size = lru_slots.shape[1]
    _need(region_stride == hot_size and hot_size >= top_k
          and fits(max_union_rows, hot_size, 1 << 30),
          f"unsupported shape (hot {hot_size}, top-k {top_k}, union {max_union_rows})")
    _need(global_indices.dtype == torch.int32 and global_indices.stride(1) == 1
          and hot_indices.is_contiguous() and device_global_indices.is_contiguous()
          and lru_slots.is_contiguous() and lru_scratch.shape == lru_slots.shape
          and lru_scratch.dtype == torch.int16 and hot_block_table.stride(1) == 1
          and (request_ids is None or (request_ids.dtype == torch.int32
                                       and request_ids.is_contiguous()))
          and (source_block_table is None or source_block_table.stride(1) == 1)
          and (resident_block_table is None or resident_block_table.stride(1) == 1)
          and (attention_indices is None or attention_indices.stride(1) == 1)
          and (swap_counts is None or (swap_host_physical_rows.is_contiguous()
                                       and swap_device_physical_rows.is_contiguous())),
          "unexpected tensor layout")
    if rows == 0 or top_k == 0:
        return
    src, res = source_block_table, resident_block_table
    params = _RESOLVE_PARAMS.pack(
        _ptr(hot_block_table), _ptr(global_indices), _ptr(request_ids), _ptr(src), _ptr(res),
        _ptr(resolved_global_indices), _ptr(valid_counts), _ptr(swap_host_physical_rows),
        _ptr(swap_device_physical_rows), _ptr(swap_counts), _ptr(hot_indices),
        _ptr(attention_indices), _ptr(miss_mask), _ptr(device_global_indices), _ptr(lru_slots),
        _ptr(lru_scratch), _ptr(stats), _ptr(request_state_indices),
        0 if request_state_indices is None else request_state_indices.numel(),
        host_cache.shape[0], hot_block_table.stride(0), hot_cache.shape[1], top_k, hot_size,
        max_union_rows + 1, region_stride, attention_block_stride,
        0 if src is None else src.stride(0), 0 if src is None else src.shape[0],
        0 if src is None else src.shape[1], source_block_size,
        0 if res is None else res.stride(0), 0 if res is None else res.shape[0],
        0 if res is None else res.shape[1], resident_block_size, resident_null_block,
        global_indices.stride(0), 0 if attention_indices is None else attention_indices.stride(0),
        0 if valid_counts is None else valid_counts.stride(0))
    _launch(global_indices.device.index, "suffix_hisparse_resolve", (rows, 1), 1024,
            smem_bytes(max_union_rows, hot_size), params)


def invalidate_written_slots(device_global_indices, request_state_indices, req_id_per_token,
                             written_slots) -> None:
    """torch.ops._C_cache_ops.hisparse_invalidate_written_slots, same arguments."""
    import torch

    _need(req_id_per_token.dtype == torch.int32 and written_slots.dtype == torch.int64
          and request_state_indices.dtype == torch.int32 and device_global_indices.is_contiguous()
          and req_id_per_token.is_contiguous() and written_slots.is_contiguous()
          and request_state_indices.is_contiguous()
          and req_id_per_token.numel() == written_slots.numel(), "unexpected invalidation inputs")
    tokens = written_slots.numel()
    rows, region = device_global_indices.shape
    if tokens == 0 or rows == 0 or region == 0:
        return
    params = _INVALIDATE_PARAMS.pack(
        _ptr(device_global_indices), _ptr(request_state_indices), _ptr(req_id_per_token),
        _ptr(written_slots), tokens, request_state_indices.numel(), rows, region)
    _launch(device_global_indices.device.index, "suffix_hisparse_invalidate",
            (tokens, -(-region // 1024)), 256, 0, params)


def _op(rt, which: str):
    """Ours, unless the runtime is an oracle's upstream reference."""
    import functools

    import torch

    if getattr(rt, "_suffix_upstream", False):
        ops = torch.ops._C_cache_ops
        return (ops.hisparse_resolve_residency if which == "resolve"
                else ops.hisparse_invalidate_written_slots)
    if which == "invalidate":
        return invalidate_written_slots
    return functools.partial(resolve_residency, lru_scratch=rt.index_group._suffix_lru_scratch)


def patch_source(src: str) -> dict:
    """Pure check of hisparse/runtime.py: {method: rewritten, dedented source} with
    the two op calls routed through _suffix_hisparse_op. Each call must occur once
    in the file, inside its HiSparseRuntime method."""
    import ast
    import textwrap

    from hisparse_mtp_patch import PatchDriftError

    cls = next((n for n in ast.parse(src).body
                if isinstance(n, ast.ClassDef) and n.name == "HiSparseRuntime"), None)
    lines = src.splitlines(keepends=True)
    methods = {n.name: "".join(lines[n.lineno - 1:n.end_lineno])
               for n in (cls.body if cls else []) if isinstance(n, ast.FunctionDef)}
    out = {}
    for name, old, new in EDITS:
        body = methods.get(name, "")
        if src.count(old) != 1 or body.count(old) != 1:
            raise PatchDriftError(f"hisparse/runtime.py: own-resolver anchor {name} drifted "
                                  f"(file {src.count(old)}x, method {body.count(old)}x)")
        out[name] = textwrap.dedent(body.replace(old, new))
        compile(out[name], f"<own-resolver:{name}>", "exec")
    return out


def install(runtime, methods: dict, rt_path) -> None:
    """Bind the rewritten methods on the live HiSparseRuntime, give every index
    group its LRU scratch and this process its kernels at __init__ (never inside
    graph capture), and check OUR shared memory instead of upstream's."""
    import __future__
    import linecache

    import torch

    cls = runtime.HiSparseRuntime
    for name, src in methods.items():
        fname = f"{rt_path}.own-resolver-{name}.py"
        linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
        ns: dict = {}
        exec(compile(src, fname, "exec", __future__.annotations.compiler_flag, dont_inherit=True),
             runtime.__dict__, ns)
        ns[name].__qualname__ = f"HiSparseRuntime.{name}"
        setattr(cls, name, ns[name])
    orig_init = cls.__init__

    def __init__(self, config, *a, **kw):
        orig_init(self, config, *a, **kw)
        if self.device.type == "cuda" and self.is_group_leader:
            self.index_group._suffix_lru_scratch = torch.empty_like(self.index_group.lru_slots)
            _kernels(self.device.index if self.device.index is not None
                     else torch.cuda.current_device())

    def check(device, config) -> None:
        index = device.index if device.index is not None else torch.accelerator.current_device_index()
        available = runtime.get_max_shared_memory_bytes(index)
        if not fits(config.max_union_rows, config.device_buffer_size, available):
            raise ValueError(
                f"own HiSparse resolver: {smem_bytes(config.max_union_rows, config.device_buffer_size)}"
                f" B of shared memory for {config.max_union_rows} union rows (hot "
                f"{config.device_buffer_size}, slots < {POS_LIMIT}); the device allows {available} B")

    __init__.__qualname__ = "HiSparseRuntime.__init__"
    cls.__init__ = __init__
    runtime._suffix_hisparse_op = _op
    runtime._check_residency_shared_memory = check


# ------------------------------------------------------------------ reference
def reference_resolve(hot_bt, gidx, req_ids, state_idx, src_bt, src_bs, res_bt, res_bs, res_null,
                      host_rows, dgi, lru, hot_bs, attn_block_stride):
    """The resolver's semantics in plain torch/Python (CPU tensors), one request
    run at a time. Returns (outputs, new dgi, new lru); inputs are not modified.
    Unwritten swap entries stay -9."""
    import torch

    dgi, lru = dgi.clone(), lru.clone()
    n, k = gidx.shape
    hot = lru.shape[1]
    out = {key: torch.full((n, k), -9, dtype=torch.int32)
           for key in ("hot", "attn", "swap_host", "swap_dev")}
    out.update(valid=torch.full((n,), -9, dtype=torch.int32),
               swap_cnt=torch.full((n,), -9, dtype=torch.int32), hits=0, misses=0)

    def attn_of(p: int) -> int:
        return -1 if p < 0 else (p // hot_bs) * attn_block_stride + p % hot_bs

    ids = [int(x) for x in req_ids]
    row = 0
    while row < n:
        end = row + 1
        while end < n and ids[end] == ids[row]:
            end += 1
        rid = ids[row]
        st = int(state_idx[rid]) if 0 <= rid < state_idx.numel() else -1
        if st < 0:  # padding: everything -1, nothing resolved
            for key, v in (("hot", -1), ("attn", -1), ("valid", 0), ("swap_cnt", 0)):
                out[key][row:end] = v
            row = end
            continue
        t = gidx[row:end].long()
        tp = t.clamp(min=0)
        sb = torch.where(t >= 0, tp // src_bs, torch.full_like(t, -1))
        ok = (0 <= rid < src_bt.shape[0]) & (sb >= 0) & (sb < src_bt.shape[1])
        pb = torch.where(ok, src_bt[min(max(rid, 0), src_bt.shape[0] - 1)][
            sb.clamp(0, src_bt.shape[1] - 1)].long(), torch.zeros_like(t))
        g = torch.where(ok & (pb > 0), pb * src_bs + tp % src_bs, torch.full_like(t, -1))
        g = torch.where(g >= host_rows, torch.full_like(g, -1), g)
        resident = torch.full_like(t, -1)
        if res_bt is not None:
            rb = torch.where(t >= 0, tp // res_bs, torch.full_like(t, -1))
            okr = (st < res_bt.shape[0]) & (rb >= 0) & (rb < res_bt.shape[1])
            prb = torch.where(okr, res_bt[min(st, res_bt.shape[0] - 1)][
                rb.clamp(0, res_bt.shape[1] - 1)].long(), torch.full_like(t, -1))
            resident = torch.where(okr & (prb != res_null) & (prb >= 0),
                                   prb * res_bs + tp % res_bs, torch.full_like(t, -1))
        out["valid"][row:end] = ((resident >= 0) | (g >= 0)).sum(1).to(torch.int32)
        refs = []  # host references (position, g) in (row, top-k) order
        for p, (r_, g_) in enumerate(zip(resident.reshape(-1).tolist(), g.reshape(-1).tolist())):
            ri, i = row + p // k, p % k
            if r_ >= 0 or g_ < 0:
                out["hot"][ri, i], out["attn"][ri, i] = (r_, attn_of(r_)) if r_ >= 0 else (-1, -1)
            else:
                refs.append((p, g_))
        first: dict = {}
        for p, g_ in refs:
            first.setdefault(g_, p)
        if not first:  # fully resident / masked: the LRU is not consulted
            out["swap_cnt"][row:end] = 0
            row = end
            continue
        dgi_row = dgi[st].tolist()
        slot_of = {}
        hits, evict = [], []
        for s in lru[st].tolist():
            c = dgi_row[s] if 0 <= s < hot else -1
            if c >= 0 and c in first:
                slot_of[c] = ("hit", s)
                hits.append(s)
            else:
                evict.append(s)
        misses = sorted((p, g_) for g_, p in first.items() if g_ not in slot_of)
        for m, (_p, g_) in enumerate(misses):  # oldest evictable first
            s = evict[m]
            out["swap_host"].view(-1)[row * k + m] = g_
            if 0 <= s < hot:
                slot_of[g_] = ("miss", s)
                out["swap_dev"].view(-1)[row * k + m] = int(hot_bt[rid, s // hot_bs]) * hot_bs + s % hot_bs
                dgi[st, s] = g_
            else:  # corrupted LRU entry: resolved invalid, re-misses later
                slot_of[g_] = ("invalid", -1)
                out["swap_dev"].view(-1)[row * k + m] = -1
        loads = {p for p, _g in misses}
        for p, g_ in refs:
            kind, s = slot_of[g_]
            ri, i = row + p // k, p % k
            if kind == "invalid":
                out["hot"][ri, i] = out["attn"][ri, i] = -1
                continue
            phys = int(hot_bt[rid, s // hot_bs]) * hot_bs + s % hot_bs
            out["hot"][ri, i], out["attn"][ri, i] = phys, attn_of(phys)
            if kind == "miss" and p in loads:
                out["misses"] += 1
            else:
                out["hits"] += 1
        for r_ in range(row, end):
            out["swap_cnt"][r_] = max(0, min(k, len(misses) - (r_ - row) * k))
        lru[st] = torch.tensor(evict[len(misses):] + evict[:len(misses)] + hits, dtype=lru.dtype)
        row = end
    return out, dgi, lru


# --------------------------------------------------------------------- oracle
def _graph_us(fn, reps: int = 20) -> float:
    """Median device us per fn(j) call, j = 0..reps-1 captured in one CUDA graph."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for j in range(3):
            fn(j)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for j in range(reps):
            fn(j)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(7):
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    return sorted(ts)[3]


def main() -> int:
    tag = f"{TAG} oracle"
    try:
        import inspect
        import types
        from pathlib import Path

        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            print(f"{tag}: not applicable (no SM120 GPU)", flush=True)
            return 3
        from vllm.platforms import current_platform

        assert current_platform.is_cuda()  # loads vllm._C_stable_libtorch: torch.ops._C_cache_ops
        from vllm.v1.hisparse import runtime as R

        import hisparse_mtp_patch as HS
        from hisparse_mtp_patch import split_oracle as SO

        rt_path = Path(R.__file__)
        rt_src = rt_path.read_text()
        if not HS.check_split_source(rt_src):
            print(f"{tag}: not applicable (no union resolver in this vLLM)", flush=True)
            return 3
        ops = torch.ops._C_cache_ops
        ok = True

        def emit(line):  # as each check completes: a crash still shows what passed
            print(f"{tag}: {line}", flush=True)

        def names(op):
            return [a.name for a in op.default._schema.arguments]

        sig = (names(ops.hisparse_resolve_residency)
               == list(inspect.signature(resolve_residency).parameters)[:-1]
               and names(ops.hisparse_invalidate_written_slots)
               == list(inspect.signature(invalidate_written_slots).parameters))
        ok &= sig
        emit(f"0 op schemas == our signatures: {sig}")
        install(R, patch_source(rt_src), rt_path)
        HS.install_split(R, auto=False)  # split only where enable_split() asks (timing)
        dev = torch.device("cuda", 0)
        torch.cuda.set_device(dev)
        optin = torch.cuda.get_device_properties(dev).shared_memory_per_block_optin
        gen = torch.Generator().manual_seed(4321)
        T, BS, HOT, UNION = SO.TOP_K, SO.BS, 7 * SO.TOP_K, 6 * SO.TOP_K  # GLM: k=5 sizing
        reqs = 3
        host, bt = SO._host(reqs, dev)

        def runtime(q, upstream=False, hot=None, union=None, swap_rows=None, reqs=reqs, host=host):
            rt, raw = SO._runtime(R, q, reqs, dev, hot=hot, union=union, swap_rows=swap_rows)
            rt.bind_source_cache(host)
            rt._suffix_upstream = upstream
            return rt, raw

        def step(rt, ids, topk, resident=None, swap=True, bt=bt):
            rt.begin_forward()
            rows = rt._step_rows(topk.shape[0])
            rt._resolve_residency(resident=resident, req_id_per_token=ids, block_table=bt,
                                  topk_indices=topk, block_size=BS, return_valid_counts=True,
                                  shared_rows=rows)
            if swap:
                rt._swap_rows(rows)
            return rows

        def state(rt):
            g = rt.index_group
            return (g.device_global_indices.clone(), g.lru_slots.clone(),
                    g.swap_stats.clone().view(torch.int64))

        def outputs(rt, rows, before):
            sh, g = rt.index_group.shared_topk, rt.index_group
            cnt = sh.swap_counts[rows]
            live = torch.arange(T, device=dev)[None, :] < cnt[:, None]
            return {"hot": sh.device_topk_rows[rows], "attn": sh.physical_topk_indices[rows],
                    "valid": sh.valid_topk_counts[rows], "swap_cnt": cnt,
                    "swap_host": torch.where(live, sh.swap_host_physical_rows[rows], -9),
                    "swap_dev": torch.where(live, sh.swap_device_physical_rows[rows], -9),
                    "dgi": g.device_global_indices, "lru": g.lru_slots,
                    "stats": g.swap_stats.view(torch.int64) - before[2]}

        def reference(rt, ids, topk, before, resident):
            ref, rdgi, rlru = reference_resolve(
                rt.hot_block_table.cpu(), topk.cpu(), ids.cpu(), rt.request_state_indices.cpu(),
                bt.cpu(), BS, None if resident is None else resident.block_table.cpu(), BS, 0,
                host.shape[0], before[0].cpu(), before[1].cpu(), BS, rt.hot.attention_block_stride)
            return {**{k: ref[k] for k in ("hot", "attn", "valid", "swap_cnt", "swap_host",
                                           "swap_dev")},
                    "dgi": rdgi, "lru": rlru, "stats": torch.tensor([ref["hits"], ref["misses"]])}

        def diff(a, b):
            return [k for k in a if not torch.equal(a[k].cpu(), b[k].cpu())]

        def case(q, j, topk):
            """Step j's launch rows: plain, a resident table, V2 padding (request -1)
            or a request-0 run of masked rows after the real ones."""
            ids = torch.arange(reqs, dtype=torch.int32, device=dev).repeat_interleave(q)
            resident = None
            if j % 4 == 1:
                tbl = torch.randint(1, reqs * HOT // BS, (reqs, bt.shape[1]), generator=gen,
                                    dtype=torch.int32)
                tbl[torch.rand(tbl.shape, generator=gen) < 0.75] = 0  # null: host-backed
                resident = types.SimpleNamespace(block_table=tbl.to(dev),
                                                 view=types.SimpleNamespace(block_size=BS))
            elif j % 4 == 2:
                ids = torch.cat([ids, torch.full((q,), -1, dtype=torch.int32, device=dev)])
                topk = torch.cat([topk, topk[:q]])
            elif j % 4 == 3:
                ids = torch.cat([ids, torch.zeros(q, dtype=torch.int32, device=dev)])
                topk = torch.cat([topk, torch.full_like(topk[:q], -1)])
            return ids, topk, resident

        # A: k=2 / k=3 sizing (upstream fits one pass): ours == upstream on every
        # output and all state, and upstream == the reference B relies on.
        for q in (3, 4):
            up, _ = runtime(q, upstream=True, swap_rows=(reqs + 1) * q)
            ours, ours_raw = runtime(q, swap_rows=(reqs + 1) * q)
            good, ref_good, first_bad = 0, 0, None
            for j in range(12):
                ids, topk, resident = case(q, j, SO._topk(gen, q, reqs, dev))
                b_up, b_ours = state(up), state(ours)
                r_up = step(up, ids, topk, resident)
                r_ours = step(ours, ids, topk, resident)
                torch.cuda.synchronize()
                o_up, o_ours = outputs(up, r_up, b_up), outputs(ours, r_ours, b_ours)
                bad = diff(o_up, o_ours)
                n = reqs * q
                if resident is None and not SO._check(ours_raw, host, bt, ids[:n], topk[:n],
                                                       o_ours["attn"][:n]):
                    bad.append("host bytes")
                good += not bad
                first_bad = first_bad or (bad and (j, bad))
                ref_good += not diff(reference(up, ids, topk, b_up, resident), o_up)
            ok &= good == 12 and ref_good == 12
            emit(f"A k={q - 1} ({q} rows/request, hot {(q + 1) * T}): ours vs upstream "
                         f"{good}/12 steps bit-identical (indices, valid counts, swaps, dgi, LRU, "
                         f"stats; incl. resident + padding runs), upstream vs reference "
                         f"{ref_good}/12" + (f"; first diff {first_bad}" if first_bad else ""))

        # B: GLM k=5 sizing (upstream needs more shared memory than SM120 has):
        # ours in ONE pass == the reference, and every row reads its host bytes.
        rt, raw = runtime(6, hot=HOT, union=UNION, swap_rows=(reqs + 1) * 6)
        good, first_bad = 0, None
        for j in range(12):
            ids, topk, resident = case(6, j, SO._topk(gen, 6, reqs, dev))
            before = state(rt)
            rows = step(rt, ids, topk, resident)
            torch.cuda.synchronize()
            got = outputs(rt, rows, before)
            bad = diff(reference(rt, ids, topk, before, resident), got)
            n = reqs * 6
            if resident is None and not SO._check(raw, host, bt, ids[:n], topk[:n],
                                                   got["attn"][:n]):
                bad.append("host bytes")
            good += not bad
            first_bad = first_bad or (bad and (j, bad))
        ok &= good == 12
        emit(f"B k=5 (hot {HOT}, union {UNION}): one pass, {smem_bytes(UNION, HOT)} B "
                     f"shared memory (upstream {HS.residency_smem_bytes(UNION, HOT)} B, device "
                     f"{optin} B): {good}/12 steps == reference, host bytes exact"
                     + (f"; first diff {first_bad}" if first_bad else ""))

        # C: the k=5 step (resolve + swap) in one CUDA graph, replayed with new top-k.
        rt, raw = runtime(6, hot=HOT, union=UNION)
        ids6 = torch.arange(reqs, dtype=torch.int32, device=dev).repeat_interleave(6)
        topk = SO._topk(gen, 6, reqs, dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step(rt, ids6, topk)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rows = step(rt, ids6, topk)
        phys = rt.index_group.shared_topk.physical_topk_indices[rows]
        good = 0
        for _ in range(8):
            topk.copy_(SO._topk(gen, 6, reqs, dev))
            graph.replay()
            torch.cuda.synchronize()
            good += SO._check(raw, host, bt, ids6, topk, phys)
        ok &= good == 8
        emit(f"C k=5 in a CUDA graph: {good}/8 replays byte-exact")

        # D: written-slot invalidation, ours vs upstream on the same random state.
        good = 0
        for _ in range(8):
            dgi = torch.randint(-1, 3000, (8, HOT), dtype=torch.int32, device=dev)
            states = torch.randperm(10, device=dev)[:6].to(torch.int32)  # >= 8: out of range
            states[1] = -1
            rid = torch.randint(-1, 7, (40,), dtype=torch.int32, device=dev)  # 6: no state
            written = torch.randint(-1, 3000, (40,), dtype=torch.int64, device=dev)
            a, b = dgi.clone(), dgi.clone()
            ops.hisparse_invalidate_written_slots(a, states, rid, written)
            invalidate_written_slots(b, states, rid, written)
            torch.cuda.synchronize()
            good += torch.equal(a, b) and not torch.equal(a, dgi)
        ok &= good == 8
        emit(f"D written-slot invalidation ours vs upstream: {good}/8 bit-identical")

        # Timing: graphed us per call, 4 top-k sets cycling (LRU churn), resolve
        # alone + one layer's swap copy. GLM sizing: ours vs the split it
        # replaces; k=2/k=3 sizing: ours vs upstream's single pass.
        for b_, q, sizing in ((1, 6, "glm"), (8, 4, "glm"), (32, 3, "glm"),
                              (8, 4, "fit"), (32, 3, "fit")):
            host_t, bt_t = SO._host(b_, dev)
            ids_t = torch.arange(b_, dtype=torch.int32, device=dev).repeat_interleave(q)
            sets = [SO._topk(gen, q, b_, dev) for _ in range(4)]
            cells = []
            for label in ("ours", "split" if sizing == "glm" else "upstream"):
                kw = dict(hot=HOT, union=UNION) if sizing == "glm" else {}
                rt_t, _raw = runtime(q, upstream=label != "ours", reqs=b_, host=host_t,
                                     swap_rows=b_ * q + 1, **kw)
                if label == "split":
                    HS.enable_split(R, rt_t, HS.residency_pass_rows(6, T, HOT, optin), 6, T)
                res_us = _graph_us(lambda j, rt_t=rt_t: step(rt_t, ids_t, sets[j % 4], swap=False,
                                                             bt=bt_t))
                rows_t = step(rt_t, ids_t, sets[0], bt=bt_t)
                swap_us = _graph_us(lambda j, rt_t=rt_t, rows_t=rows_t: rt_t._swap_rows(rows_t))
                cells.append(f"{label} {res_us:6.1f} + {swap_us:5.1f}")
                del rt_t, _raw
            hot_t = HOT if sizing == "glm" else (q + 1) * T
            emit(f"timing batch {b_:2d} x {q} rows (hot {hot_t}), resolve + swap us: "
                         + " | ".join(cells))
            torch.cuda.empty_cache()
        dgi = torch.randint(-1, 3000, (32, HOT), dtype=torch.int32, device=dev)
        states = torch.arange(32, dtype=torch.int32, device=dev)
        for ntok in (6, 32, 96):
            rid = torch.arange(ntok, dtype=torch.int32, device=dev) % 32
            written = torch.randint(0, 3000, (ntok,), dtype=torch.int64, device=dev)
            up_us = _graph_us(lambda j: ops.hisparse_invalidate_written_slots(dgi, states, rid, written))
            our_us = _graph_us(lambda j: invalidate_written_slots(dgi, states, rid, written))
            emit(f"timing invalidate {ntok:3d} tokens (hot {HOT}): ours {our_us:5.1f} | "
                         f"upstream {up_us:5.1f} us")
    except Exception as exc:  # noqa: BLE001 - any error is a FAIL: refuse to serve
        import traceback

        print(f"{tag}: ERROR {exc!r}\n{traceback.format_exc()[-2000:]}", flush=True)
        return 1
    print(f"{tag}: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
