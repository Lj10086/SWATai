"""Parameter table writes — especially integer columns that gfortran mis-reads as floats."""
from __future__ import annotations


from swat_gym.fastrunner import TXTINOUT
from swat_gym.params import get_value, set_value


def test_days_mat_written_as_integer():
    text = (TXTINOUT / "plants.plt").read_text()
    out = set_value(text, "plants.plt", "corn", "days_mat", 120.0)
    assert get_value(out, "plants.plt", "corn", "days_mat") == 120.0
    # Token must lack a decimal point — ``120.00000`` shifts later columns under rev 62.
    for line in out.splitlines():
        if line.split()[:1] == ["corn"]:
            hdr = out.splitlines()[1].split()
            tok = line.split()[hdr.index("days_mat")]
            assert tok == "120", tok
            break
    else:
        raise AssertionError("corn row missing")


def test_bm_e_still_five_decimals():
    text = (TXTINOUT / "plants.plt").read_text()
    out = set_value(text, "plants.plt", "corn", "bm_e", 50.0)
    for line in out.splitlines():
        if line.split()[:1] == ["corn"]:
            hdr = out.splitlines()[1].split()
            assert line.split()[hdr.index("bm_e")] == "50.00000"
            break
