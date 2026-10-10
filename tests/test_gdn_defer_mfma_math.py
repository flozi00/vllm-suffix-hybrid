# SPDX-License-Identifier: Apache-2.0
"""CPU check (numpy fp64) of the algebra in gdn_defer_rocm._gdn_defer_mfma_kernel: the
chunk form over the kernel's 16-column layout (0..3 replayed tokens, 4..8 new k, 9..13
new q; masked columns are identity tokens), the (I + A)^-1 by doubling, outputs and the
states after any column, chained over random acceptance under the deferred contract
(slot 0 + records, replay) and the stock one (a state per token). Yardstick: the
sequential recurrence of _gdn_token. The kernel itself is checked on silicon: boot gate
gdn_defer_mfma_bench."""
import numpy as np

K, V, WIN, NREP = 128, 64, 5, 4
COLS = np.arange(16)
IS_Q = (COLS >= NREP + WIN) & (COLS < NREP + 2 * WIN)


def _gate(a_raw, b_raw, A_log, dt):
    x = a_raw + dt
    sp = np.where(x <= 20.0, np.log(1 + np.exp(np.minimum(x, 20.0))), x)
    return -np.exp(A_log) * sp, 1 / (1 + np.exp(-b_raw))


def seq_token(h, k_raw, v, a_raw, b_raw, A_log, dt, q_raw=None):
    """_gdn_token + the output, fp64."""
    g, beta = _gate(a_raw, b_raw, A_log, dt)
    k = k_raw / np.sqrt((k_raw * k_raw).sum() + 1e-6)
    h = h * np.exp(g)
    u = (v - h @ k) * beta
    h = h + np.outer(u, k)
    if q_raw is None:
        return h, None
    q = q_raw / np.sqrt((q_raw * q_raw).sum() + 1e-6) * K**-0.5
    return h, h @ q


def chunk(S0, kq_raw, vt, a_c, b_c, use, A_log, dt):
    """The kernel's per-program math. kq_raw [16, K] raw k (cols 0..8) / raw q (9..13),
    vt [V, 16], a_c / b_c / use [16] (use: valid k columns). Returns outputs [V, 16]
    (q columns) and state_after(c)."""
    nrm = 1 / np.sqrt((kq_raw * kq_raw).sum(1) + 1e-6)
    KQ = kq_raw * np.where(IS_Q, nrm * K**-0.5, nrm)[:, None]
    g, beta = _gate(a_c, b_c, A_log, dt)
    g, beta = np.where(use, g, 0.0), np.where(use, beta, 0.0)
    G = np.cumsum(g)
    WZ = S0 @ KQ.T  # [V, 16]: S0 k_c / S0 q_j
    GR = KQ @ KQ.T  # [16, 16]
    t, s = COLS[:, None], COLS[None, :]
    A = np.where((s < t) & (t < NREP + WIN),
                 beta[:, None] * np.exp(np.minimum(G[:, None] - G[None, :], 0.0)) * GR, 0.0)
    X, I = -A, np.eye(16)
    X2 = X @ X
    X4 = X2 @ X2
    P = (I + X) @ (I + X2) @ (I + X4) @ (I + X4 @ X4)  # (I + A)^-1: A^16 = 0
    Gam = np.exp(G)
    U = (beta[None, :] * (vt - Gam[None, :] * WZ)) @ P.T
    # q column j belongs to chunk column j - WIN; its decay G[j - WIN] by a selection sum
    Gq = (G[:, None] * ((s == t + WIN) & IS_Q[None, :])).sum(0)
    Bm = np.where(IS_Q[None, :] & (t <= s - WIN),
                  np.exp(np.minimum(Gq[None, :] - G[:, None], 0.0)) * GR, 0.0)
    O = WZ * np.where(IS_Q, np.exp(Gq), 0.0)[None, :] + U @ Bm

    def state_after(c):
        d = np.where(COLS <= c, np.exp(np.minimum(G[c] - G, 0.0)), 0.0)
        return Gam[c] * S0 + (U * d[None, :]) @ KQ

    return O, state_after


def _run(zone_mode, steps=40, seed=0):
    """One request through `steps` MTP steps. zone_mode False: deferred (slot 0 + records
    of tokens 1.., replay acc - 1 of them); True: stock way (a state per token, read slot
    acc - 1, no replay). Returns max abs errors (committed state, outputs) vs fp64 seq."""
    rng = np.random.default_rng(seed)
    A_log, dt = 0.5 * rng.standard_normal(), 0.5 * rng.standard_normal()
    E = 0.1 * rng.standard_normal((V, K))  # sequential committed state
    slots = [E.copy()] + [None] * (WIN - 1)
    recs, acc, err_s, err_o = [], 1, 0.0, 0.0
    for _ in range(steps):
        n = int(rng.integers(1, WIN + 1))
        k_raw, q_raw = rng.standard_normal((n, K)), rng.standard_normal((n, K))
        v, a, b = rng.standard_normal((n, V)), rng.standard_normal(n), rng.standard_normal(n)
        # fp64 sequential: outputs of the new tokens from the committed state E
        h, outs, states = E, [], []
        for i in range(n):
            h, o = seq_token(h, k_raw[i], v[i], a[i], b[i], A_log, dt, q_raw[i])
            outs.append(o)
            states.append(h)
        kq, vt = np.zeros((16, K)), np.zeros((V, 16))
        a_c, b_c, use = np.zeros(16), np.zeros(16), np.zeros(16, bool)
        deferred = not zone_mode and acc >= 2
        S0 = slots[0] if deferred else slots[acc - 1]
        if deferred:
            for c in range(acc - 1):  # record t = c + 1 of the previous step
                kq[c], vt[:, c], a_c[c], b_c[c] = recs[c]
                use[c] = True
        kq[NREP:NREP + n], vt[:, NREP:NREP + n] = k_raw, v.T
        kq[NREP + WIN:NREP + WIN + n] = q_raw
        a_c[NREP:NREP + n], b_c[NREP:NREP + n], use[NREP:NREP + n] = a, b, True
        O, state_after = chunk(S0, kq, vt, a_c, b_c, use, A_log, dt)
        err_o = max(err_o, np.abs(O[:, NREP + WIN:NREP + WIN + n].T - np.array(outs)).max())
        slots = [state_after(NREP)] + [None] * (WIN - 1)
        if zone_mode:
            for i in range(1, n):
                slots[i] = state_after(NREP + i)
        recs = [(k_raw[i], v[i], a[i], b[i]) for i in range(1, n)]
        acc = int(rng.integers(1, n + 1))
        E = states[acc - 1]
        # the committed state as the next step will rebuild it
        if zone_mode or acc == 1:
            got = slots[acc - 1]
        else:
            h = slots[0]
            for c in range(acc - 1):
                h, _ = seq_token(h, *recs[c], A_log, dt)
            got = h
        err_s = max(err_s, np.abs(got - E).max())
    return err_s, err_o


def test_chunk_form_matches_sequential_recurrence():
    for zone_mode in (False, True):
        for seed in range(4):
            err_s, err_o = _run(zone_mode, seed=seed)
            assert err_s < 1e-12 and err_o < 1e-12, (zone_mode, seed, err_s, err_o)


def test_replay_inside_the_chunk():
    # r replayed + 5 new tokens with nearly parallel keys (k_s . k_t ~ 1, the long
    # chains of (I + A)^-1 matter): every state and output == the sequential recurrence.
    rng = np.random.default_rng(7)
    A_log, dt = -2.0, 2.0
    S0 = 0.1 * rng.standard_normal((V, K))
    base = rng.standard_normal(K)
    for r in range(NREP + 1):
        kq, vt = np.zeros((16, K)), np.zeros((V, 16))
        a_c, b_c, use = np.zeros(16), np.zeros(16), np.zeros(16, bool)
        cols = list(range(r)) + list(range(NREP, NREP + WIN))
        h, states, outs = S0, {}, {}
        for c in cols:
            kq[c] = base + 0.05 * rng.standard_normal(K)
            vt[:, c], a_c[c], b_c[c], use[c] = rng.standard_normal(V), rng.standard_normal(), 3.0, True
            q = kq[c + WIN] = rng.standard_normal(K) if c >= NREP else None
            h, outs[c] = seq_token(h, kq[c], vt[:, c], a_c[c], b_c[c], A_log, dt, q)
            states[c] = h
        O, state_after = chunk(S0, kq, vt, a_c, b_c, use, A_log, dt)
        for c in cols:
            assert np.abs(state_after(c) - states[c]).max() < 1e-12, (r, c)
            if c >= NREP:
                assert np.abs(O[:, c + WIN] - outs[c]).max() < 1e-12, (r, c)


if __name__ == "__main__":
    for zm in (False, True):
        print("zone_mode" if zm else "deferred", [_run(zm, seed=s) for s in range(4)])
    test_replay_inside_the_chunk()
    print("ok")
