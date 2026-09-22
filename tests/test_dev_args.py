"""dev_args splice semantics: override-in-place, append-new, remove, guardrails."""
import os

# No sys.path manipulation: the package resolves via the installed wheel in
# CI and via pytest's rootdir insertion locally. Inserting the repo root
# here would shadow the wheel with the source dir (no compiled .so) and
# break every native test module that imports after this one.
from suffix_hybrid.dev_args import splice  # noqa: E402

BASE = ["vllm", "serve", "s3://model-weights/m/", "--load-format",
        "runai_streamer", "--host", "0.0.0.0", "--port", "8000",
        "--speculative-config.num_speculative_tokens", "12",
        "--moe-backend", "marlin", "--enable-expert-parallel"]


def test_override_keeps_position():
    out = splice(BASE, [["--moe-backend", "flashinfer_b12x"]])
    assert out.count("--moe-backend") == 1
    assert out[out.index("--moe-backend") + 1] == "flashinfer_b12x"
    # position preserved: still before --enable-expert-parallel
    assert out.index("--moe-backend") < out.index("--enable-expert-parallel")


def test_new_flag_appends():
    out = splice(BASE, [["--max-num-seqs", "24"]])
    assert out[-2:] == ["--max-num-seqs", "24"]


def test_bare_flag_appends_once():
    out = splice(BASE, ["--enable-prefix-caching", "--enable-prefix-caching"])
    assert out.count("--enable-prefix-caching") == 1


def test_removal():
    out = splice(BASE, [["--moe-backend", None]])
    assert "--moe-backend" not in out and "marlin" not in out


def test_valueless_flag_override():
    out = splice(BASE, [["--enable-expert-parallel", None],
                        "--enable-expert-parallel"])
    assert out.count("--enable-expert-parallel") == 1


def test_equals_form_parses_and_overrides():
    argv = BASE + ["--gpu-memory-utilization=0.9"]
    out = splice(argv, [["--gpu-memory-utilization", "0.8"]])
    assert "--gpu-memory-utilization=0.9" not in out
    assert out[out.index("--gpu-memory-utilization") + 1] == "0.8"


def test_reserved_flags_rejected():
    for flag in ("--model", "--served-model-name", "--model-path", "-m"):
        try:
            splice(BASE, [[flag, "evil"]])
            raise AssertionError(f"{flag} must be rejected")
        except ValueError:
            pass


def test_bad_entries_rejected():
    for entry in ("nope", ["--x"], ["--x", 1, 2], 42):
        try:
            splice(BASE, [entry])
            raise AssertionError(f"{entry!r} must be rejected")
        except ValueError:
            pass


def test_positional_model_untouched():
    out = splice(BASE, [["--max-num-seqs", "8"]])
    assert out[2] == "s3://model-weights/m/"


def test_gate_off_is_noop(monkeypatch=None):
    import importlib
    import suffix_hybrid.dev_args as da
    saved = os.environ.pop("SUFFIX_HYBRID_DEV_ARGS", None)
    try:
        argv = list(BASE)
        assert da.apply_dev_args(argv) is None
        assert argv == BASE
    finally:
        if saved is not None:
            os.environ["SUFFIX_HYBRID_DEV_ARGS"] = saved
        importlib.reload(da)


def test_gate_on_but_not_serve_entrypoint_is_noop():
    import suffix_hybrid.dev_args as da
    saved = os.environ.get("SUFFIX_HYBRID_DEV_ARGS")
    os.environ["SUFFIX_HYBRID_DEV_ARGS"] = "1"
    try:
        argv = ["python3", "/plugins/whatever.py"]
        assert da.apply_dev_args(argv) is None
        assert argv == ["python3", "/plugins/whatever.py"]
    finally:
        if saved is None:
            os.environ.pop("SUFFIX_HYBRID_DEV_ARGS", None)
        else:
            os.environ["SUFFIX_HYBRID_DEV_ARGS"] = saved
