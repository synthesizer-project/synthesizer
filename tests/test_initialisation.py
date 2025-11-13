"""Tests for the initialisation module."""

import os
import shutil
from importlib import resources
from io import BytesIO
from pathlib import Path

import pytest

import synthesizer.data.initialise as init_mod
from synthesizer.data.initialise import (
    SynthesizerInitializer,
    base_dir_exists,
    data_dir_exists,
    get_base_dir,
    get_data_dir,
    get_grids_dir,
    get_instrument_dir,
    get_test_data_dir,
    grids_dir_exists,
    instrument_cache_exists,
    synth_clear_data,
    synth_initialise,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure no Synthesizer‐specific env vars leak into tests."""
    for var in [
        "SYNTHESIZER_DIR",
        "SYNTHESIZER_DATA_DIR",
        "SYNTHESIZER_GRID_DIR",
        "SYNTHESIZER_TEST_DATA_DIR",
        "SYNTHESIZER_INSTRUMENT_CACHE",
    ]:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in [
        "SYNTHESIZER_DIR",
        "SYNTHESIZER_DATA_DIR",
        "SYNTHESIZER_GRID_DIR",
        "SYNTHESIZER_TEST_DATA_DIR",
        "SYNTHESIZER_INSTRUMENT_CACHE",
    ]:
        monkeypatch.delenv(var, raising=False)


class TestEnvAndPaths:
    """Tests for environment variables and directory path functions."""

    def test_get_base_dir_env_override(self, monkeypatch, tmp_path):
        """Test that environment variable overrides default base dir."""
        monkeypatch.setenv("SYNTHESIZER_DIR", str(tmp_path / "foo"))
        assert get_base_dir() == tmp_path / "foo"

    def test_get_base_dir_default(self, monkeypatch, tmp_path):
        """Test that default base dir is used if no env var set."""
        # monkeypatch user_data_dir to return our tmp_path
        monkeypatch.setattr(
            init_mod, "user_data_dir", lambda name: str(tmp_path / name)
        )
        assert get_base_dir() == tmp_path / "Synthesizer"

    def test_other_dirs_env_and_defaults(self, monkeypatch, tmp_path):
        """Test that other directories use base dir or env vars."""
        # base dir default
        monkeypatch.setattr(
            init_mod, "user_data_dir", lambda name: str(tmp_path / name)
        )
        base = get_base_dir()

        # data dir
        assert get_data_dir() == base / "data", (
            "Default data dir should be base/data"
        )
        monkeypatch.setenv("SYNTHESIZER_DATA_DIR", "/custom/data")
        assert get_data_dir() == Path("/custom/data"), (
            "Custom data dir should override default"
        )

        # grids
        assert get_grids_dir() == base / "grids", (
            "Default grids dir should be base/grids"
        )
        monkeypatch.setenv("SYNTHESIZER_GRID_DIR", "/custom/grids")
        assert get_grids_dir() == Path("/custom/grids"), (
            "Custom grids dir should override default"
        )

        # test data should hang off the overridden DATA_DIR
        assert get_test_data_dir() == Path("/custom/data") / "test_data", (
            "Test data dir should be under custom data dir"
        )
        monkeypatch.setenv("SYNTHESIZER_TEST_DATA_DIR", "/custom/test")
        assert get_test_data_dir() == Path("/custom/test"), (
            "Custom test data dir should override default"
        )

        # instrument cache
        assert get_instrument_dir() == base / "instrument_cache", (
            "Default instrument cache should be base/instrument_cache"
        )
        monkeypatch.setenv("SYNTHESIZER_INSTRUMENT_CACHE", "/custom/cache")
        assert get_instrument_dir() == Path("/custom/cache"), (
            "Custom instrument cache should override default"
        )

    def test_exists_functions(self, monkeypatch, tmp_path):
        """Test that existence functions check correct paths."""
        # patch all getters to tmp_path subdirs
        monkeypatch.setattr(
            init_mod, "get_base_dir", lambda: tmp_path / "base"
        )
        monkeypatch.setattr(
            init_mod, "get_data_dir", lambda: tmp_path / "data"
        )
        monkeypatch.setattr(
            init_mod, "get_grids_dir", lambda: tmp_path / "grids"
        )
        monkeypatch.setattr(
            init_mod, "get_test_data_dir", lambda: tmp_path / "test"
        )
        monkeypatch.setattr(
            init_mod, "get_instrument_dir", lambda: tmp_path / "inst"
        )

        # Import here to avoid the testdata directory exists function from
        # being treated a test
        from synthesizer.data.initialise import (
            testdata_dir_exists,
        )

        # none exist initially
        assert not base_dir_exists()
        assert not data_dir_exists()
        assert not grids_dir_exists()
        assert not testdata_dir_exists()
        assert not instrument_cache_exists()
        # create them
        for d in ("base", "data", "grids", "test", "inst"):
            (tmp_path / d).mkdir()
        assert base_dir_exists()
        assert data_dir_exists()
        assert grids_dir_exists()
        assert testdata_dir_exists()
        assert instrument_cache_exists()


class DummyResource(BytesIO):
    """Dummy resource to simulate resources.open_binary behavior."""

    def __enter__(self):
        """Dummy context manager enter."""
        return self

    def __exit__(self, *args):
        """Dummy context manager exit."""
        pass


class TestInitializerMethods:
    """Tests for SynthesizerInitializer methods."""

    @pytest.fixture(autouse=True)
    def patch_paths(self, monkeypatch, tmp_path):
        """Patch paths to use tmp_path for all Synthesizer dirs."""
        """Redirect all Synthesizer dirs into tmp_path."""
        monkeypatch.setenv("SYNTHESIZER_DIR", str(tmp_path / "base"))
        monkeypatch.setenv(
            "SYNTHESIZER_DATA_DIR", str(tmp_path / "base" / "data")
        )
        monkeypatch.setenv(
            "SYNTHESIZER_GRID_DIR", str(tmp_path / "base" / "grids")
        )
        monkeypatch.setenv(
            "SYNTHESIZER_TEST_DATA_DIR",
            str(tmp_path / "base" / "data" / "test"),
        )
        monkeypatch.setenv(
            "SYNTHESIZER_INSTRUMENT_CACHE", str(tmp_path / "base" / "inst")
        )
        monkeypatch.setattr(init_mod, "resources", resources)
        yield
        shutil.rmtree(tmp_path, ignore_errors=True)

    def test_make_dir(self, tmp_path):
        """Test _make_dir creates, recognizes, or fails on directories."""
        init = SynthesizerInitializer()

        # new dir
        newd = tmp_path / "foo"
        assert not newd.exists()
        init._make_dir(newd, "foo")
        assert init.status["foo"] == "created"

        # existing dir
        init._make_dir(newd, "foo")
        assert init.status["foo"] == "exists"

        # failure: use a fake path object whose mkdir() always raises
        class FakePath:
            def exists(self):
                return False

            def mkdir(self, *args, **kwargs):
                raise OSError("boom")

        fake = FakePath()
        init._make_dir(fake, "bar")
        assert init.status["bar"] == "failed"

    def test_copy_resource(self, tmp_path, monkeypatch):
        """Test _copy_resource copies resources correctly."""
        init = SynthesizerInitializer()
        dest = tmp_path / "out.yml"
        # simulate resource exists
        data = b"hello"

        def fake_open_binary(pkg, name):
            return DummyResource(data)

        monkeypatch.setattr(
            init_mod.resources, "open_binary", fake_open_binary
        )
        # first copy
        init._copy_resource("pkg", "res", dest, "res_key")
        assert init.status["res_key"] == "created"
        assert dest.read_bytes() == data
        # existing: should mark exists and not overwrite
        dest.write_bytes(b"other")
        init._copy_resource("pkg", "res", dest, "res_key")
        assert init.status["res_key"] == "exists"
        assert dest.read_bytes() == b"other"

        # failure: make open_binary throw
        def bad_open(*args):
            raise IOError

        monkeypatch.setattr(init_mod.resources, "open_binary", bad_open)
        init._copy_resource("pkg", "res", tmp_path / "new.yml", "res_key2")
        assert init.status["res_key2"] == "failed"

    def test_initialize_creates_all(self, monkeypatch):
        """Test initialize() makes all dirs and copies resources."""
        # stub out resource copy
        monkeypatch.setattr(
            init_mod.resources,
            "open_binary",
            lambda pkg, name: DummyResource(b""),
        )

        init = SynthesizerInitializer()
        init.initialize()

        # verify each real directory attribute exists and was flagged
        for attr, key in [
            ("base_dir", "base_dir"),
            ("data_dir", "data_dir"),
            ("grids_dir", "grids"),
            ("instrument_cache_dir", "instrument_cache"),
            ("test_data_dir", "test_data"),
        ]:
            path = getattr(init, attr)
            assert path.exists()
            assert init.status[key] in {"created", "exists"}

        # verify files copied
        assert (init.base_dir / "default_units.yml").exists()
        assert init.status["units_file"] in {"created", "exists"}

    def test_report_prints(self, capsys, monkeypatch):
        """Test report() prints status messages."""
        # minimal test to exercise report() without errors
        init = SynthesizerInitializer()
        # artificially fill statuses
        for k in init.status:
            init.status[k] = "exists"
        # ensure all paths exist
        for attr in (
            "base_dir",
            "data_dir",
            "grids_dir",
            "test_data_dir",
            "instrument_cache_dir",
        ):
            p = getattr(init, attr)
            p.mkdir(parents=True, exist_ok=True)
        init.report()
        out = capsys.readouterr().out
        assert "Synthesizer initialising" in out
        assert "Initialised Synthesizer directories" in out
        assert "Synthesizer initialisation complete" in out


class TestTopLevelFlows:
    """Tests for top-level functions that use SynthesizerInitializer."""

    def test_synth_initialise_skips_if_exists(self, monkeypatch, tmp_path):
        """Test synth_initialise does nothing if dirs already exist."""
        # patch all existence to True
        monkeypatch.setenv("SYNTHESIZER_DIR", str(tmp_path / "base"))
        monkeypatch.setenv(
            "SYNTHESIZER_DATA_DIR", str(tmp_path / "base" / "data")
        )
        monkeypatch.setenv(
            "SYNTHESIZER_GRID_DIR", str(tmp_path / "base" / "grids")
        )
        monkeypatch.setenv(
            "SYNTHESIZER_TEST_DATA_DIR",
            str(tmp_path / "base" / "data" / "test"),
        )
        monkeypatch.setenv(
            "SYNTHESIZER_INSTRUMENT_CACHE", str(tmp_path / "base" / "inst")
        )
        # create everything including files
        base = Path(os.environ["SYNTHESIZER_DIR"])
        (base / "data").mkdir(parents=True)
        (base / "grids").mkdir()
        (base / "data" / "test").mkdir(parents=True)
        (base / "inst").mkdir()
        # default_units.yml
        (base / "default_units.yml").write_text("")
        # should return immediately and not error
        synth_initialise()

    def test_synth_initialise_runs(self, monkeypatch, tmp_path, capsys):
        """Test synth_initialise creates dirs and files."""
        # patch env into tmp
        monkeypatch.setenv("SYNTHESIZER_DIR", str(tmp_path / "base"))
        monkeypatch.setenv(
            "SYNTHESIZER_DATA_DIR", str(tmp_path / "base" / "data")
        )
        monkeypatch.setenv(
            "SYNTHESIZER_GRID_DIR", str(tmp_path / "base" / "grids")
        )
        monkeypatch.setenv(
            "SYNTHESIZER_TEST_DATA_DIR",
            str(tmp_path / "base" / "data" / "test"),
        )
        monkeypatch.setenv(
            "SYNTHESIZER_INSTRUMENT_CACHE", str(tmp_path / "base" / "inst")
        )
        # patch resource copy
        monkeypatch.setattr(
            init_mod.resources,
            "open_binary",
            lambda pkg, name: DummyResource(b""),
        )
        synth_initialise()
        out = capsys.readouterr().out
        assert "Synthesizer initialising" in out
        assert (tmp_path / "base").exists()

    def test_synth_clear_data_removes_all(self, monkeypatch, tmp_path):
        """Test synth_clear_data removes all Synthesizer dirs."""
        # set up some dirs and files
        monkeypatch.setenv("SYNTHESIZER_DIR", str(tmp_path / "base"))
        monkeypatch.setenv(
            "SYNTHESIZER_DATA_DIR", str(tmp_path / "base" / "data")
        )
        monkeypatch.setenv(
            "SYNTHESIZER_GRID_DIR", str(tmp_path / "base" / "grids")
        )
        monkeypatch.setenv(
            "SYNTHESIZER_TEST_DATA_DIR",
            str(tmp_path / "base" / "data" / "test"),
        )
        monkeypatch.setenv(
            "SYNTHESIZER_INSTRUMENT_CACHE", str(tmp_path / "base" / "inst")
        )
        base = Path(os.environ["SYNTHESIZER_DIR"])
        data = Path(os.environ["SYNTHESIZER_DATA_DIR"])
        grids = Path(os.environ["SYNTHESIZER_GRID_DIR"])
        inst = Path(os.environ["SYNTHESIZER_INSTRUMENT_CACHE"])
        testd = Path(os.environ["SYNTHESIZER_TEST_DATA_DIR"])
        for p in (base, data, grids, inst, testd):
            p.mkdir(parents=True, exist_ok=True)
            (p / "f.txt").write_text("x")
        synth_clear_data()
        # none should exist anymore
        for p in (base, data, grids, inst, testd):
            assert not p.exists()
