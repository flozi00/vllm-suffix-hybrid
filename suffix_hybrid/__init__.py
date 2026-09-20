"""suffix_hybrid: pure-Python vLLM speculative decoding plugin.

Combines a self-improving cross-request suffix cache with vLLM's native
n-gram drafting (and an optional opt-in wrap-mode for eagle-family drafters).
"""

__version__ = "0.1.0"