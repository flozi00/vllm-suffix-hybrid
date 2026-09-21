# SPDX-License-Identifier: Apache-2.0
"""pytest bootstrap: pin the Rust suffix-cache n-gram order for unit tests.

The compiled default is n=8 (production value from SUFFIX_HYBRID_INDEX_N).
Fixtures across the suite assume short-n lookups (e.g. [1,2,3,10,11] ->
[12,99] at n=2). Setting the env var HERE -- before any test module is
imported -- makes every HybridMixer/SuffixCache in the test process use n=2
deterministically, regardless of pytest file order. Production code never
reads this file; the serving path keeps n=8 unless the operator sets
SUFFIX_HYBRID_INDEX_N explicitly.
"""
import os

os.environ.setdefault("SUFFIX_HYBRID_INDEX_N", "2")
