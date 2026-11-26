"""Build cache of the ham_dp_c library: keyed by every input of the build."""
import shutil

import pytest

from hamiltonian.claude import ham_dp_c as m


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HAM_DP_CACHE", str(tmp_path / "cache"))
    m._LIB_CACHE.clear()
    yield tmp_path / "cache"
    m.set_compiler(None)
    m.set_c_variant("key16")
    m._LIB_CACHE.clear()


def test_key_depends_on_inputs(cache):
    flags = m._build_flags({"EH_BKT_CAP": 32, "RSORT_BITS": 10})
    k1, meta = m._build_key("gcc", flags)
    assert m._build_key("gcc", flags)[0] == k1                       # deterministic
    assert m._build_key("gcc", flags + ["-DX=1"])[0] != k1           # flags
    if shutil.which("clang"):
        assert m._build_key("clang", flags)[0] != k1                 # compiler
    m.set_c_variant("legacy15")
    assert m._build_key("gcc", flags)[0] != k1                       # source/variant
    m.set_c_variant("key16")
    assert meta["source_sha256"] and meta["compiler_version"]


def test_build_reuse_list_clear(cache, capsys):
    m._get_lib()
    out = capsys.readouterr().out
    assert "ham_dp_c: compiled" in out
    m._LIB_CACHE.clear()
    m._get_lib()
    assert "ham_dp_c: cached" in capsys.readouterr().out
    m.force_rebuild()
    m._get_lib()
    assert "ham_dp_c: compiled" in capsys.readouterr().out
    m._get_lib()                                  # a forced rebuild happens once
    assert "compiled" not in capsys.readouterr().out
    entries = m.list_cache()
    assert len(entries) == 1 and entries[0][1]["variant"] == "key16"
    assert not list(cache.glob("*/*.tmp.*"))      # no half-built files left
    assert m.clear_cache() >= 1 and m.list_cache() == []
