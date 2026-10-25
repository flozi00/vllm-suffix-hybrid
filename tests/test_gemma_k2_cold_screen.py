"""A small active window must spill L2 without exhausting the full-cache allocation."""

from suffix_hybrid.tools.gemma_k2_cold_screen import copies_for


def test_small_active_window_spills_l2_despite_large_full_context():
    touched, allocated, l2 = 2 << 20, 20 << 20, 96 << 20
    copies = copies_for(touched, allocated, l2)
    assert copies * touched >= 3 * l2
    assert copies * allocated <= 6 << 30
    assert copies > 64


def test_full_allocation_budget_overrides_spill_target():
    assert copies_for(1 << 20, 2 << 30, 96 << 20) == 3
    assert copies_for(1 << 20, 8 << 30, 96 << 20) == 1


def test_supervisor_starts_isolated_child_and_returns_status(monkeypatch):
    import subprocess
    from suffix_hybrid.tools import gemma_k2_cold_screen as screen

    calls = []

    class Process:
        pid = 999999

        def wait(self, timeout):
            calls.append(timeout)
            return 0

    def start(argv, **kw):
        assert argv[-1] == "--worker" and kw["start_new_session"] is True
        return Process()

    monkeypatch.setattr(subprocess, "Popen", start)
    assert screen.main() == 0
    assert calls == [300]


def test_supervisor_kills_only_its_group_after_deadline(monkeypatch):
    import os
    import signal
    import subprocess
    from suffix_hybrid.tools import gemma_k2_cold_screen as screen

    kills = []

    class Process:
        pid = 999999
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("test", timeout)
            assert timeout == 5
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    assert screen.main() == 1
    assert kills == [(999999, signal.SIGKILL)]
