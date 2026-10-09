# SPDX-License-Identifier: Apache-2.0
"""CPU test for suffix_hybrid.gpu_mon.sample on a fake amdgpu hwmon dir."""
from suffix_hybrid import gpu_mon


def test_sample_formats_hwmon(tmp_path):
    for name, val in {"name": "amdgpu", "temp1_label": "edge", "temp1_input": "65000",
                      "temp2_label": "junction", "temp2_input": "88000",
                      "power1_average": "550000000", "power1_cap": "600000000",
                      "freq1_label": "sclk", "freq1_input": "2100000000"}.items():
        (tmp_path / name).write_text(val + "\n")
    assert gpu_mon.sample(str(tmp_path)) == "edge 65C junction 88C power 550W cap 600W sclk 2100MHz"


def test_unset_is_inert(monkeypatch):
    monkeypatch.delenv(gpu_mon.ENV, raising=False)
    assert gpu_mon.start() is False
