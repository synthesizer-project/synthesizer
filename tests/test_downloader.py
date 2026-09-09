"""Tests for resolving and downloading data files.

These cover resolution through the Synthesizer data service and verified file
downloads. No network access happens; requests is monkeypatched.
"""

import hashlib
import os
import sys

import pytest

from synthesizer import exceptions
from synthesizer.downloader import downloader

# Real payloads start with their format's signature, and the downloader now
# checks that, so the fixture has to look like the HDF5 file it claims to be.
PAYLOAD = b"\x89HDF\r\n\x1a\n" + b"some grid bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    """A stand-in for a requests response."""

    def __init__(
        self, status_code=200, payload=None, json_data=None, headers=None
    ):
        """Store what this response should return.

        Args:
            status_code (int): The HTTP status to report.
            payload (bytes): The body to stream, if any.
            json_data (dict): The body to decode as JSON, if any.
            headers (dict): The response headers.
        """
        self.status_code = status_code
        self._payload = payload or b""
        self._json = json_data
        self.headers = headers or {}

    def json(self):
        """Return the decoded body."""
        if isinstance(self._json, ValueError):
            raise self._json
        return self._json

    def iter_content(self, block_size):
        """Yield the body in chunks.

        Args:
            block_size (int): The number of bytes per chunk.
        """
        for start in range(0, len(self._payload), block_size):
            yield self._payload[start : start + block_size]


def release(sha256=DIGEST, size=len(PAYLOAD), base=None):
    """Build a catalogue response for one published dataset.

    Args:
        sha256 (str): The digest the catalogue should report.
        size (int): The file size the catalogue should report.
        base (str): The host that answered. The real API derives the download
            url from the request origin, so a fallback host serves its own.
    """
    base = base or downloader.DATA_API_URL
    return {
        "current_release": {
            "release_id": 2,
            "download_url": f"{base}/v1/releases/2/download",
            "file": {
                "filename": "test_grid.hdf5",
                "sha256": sha256,
                "size_bytes": size,
            },
        }
    }


@pytest.fixture
def fake_requests(monkeypatch):
    """Record requests and serve queued responses.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "/v1/datasets/" in url:
            return FakeResponse(json_data=fake_get.catalogue)
        return FakeResponse(
            payload=PAYLOAD, headers={"content-length": str(len(PAYLOAD))}
        )

    fake_get.catalogue = release()
    monkeypatch.setattr(downloader.requests, "get", fake_get)
    return fake_get, calls


def test_migrated_file_resolves_through_the_api(tmp_path, fake_requests):
    """A file with a dataset key is resolved and verified through the API."""
    fake_get, calls = fake_requests

    downloader._download("test_grid.hdf5", str(tmp_path))

    # The alias resolved through the data service to its catalogue name.
    assert calls[0].startswith(f"{downloader.DATA_API_URL}/v1/datasets/")
    assert calls[0].endswith(
        "/v1/datasets/bpass-2p2p1-bin-chabrier03-0p1-300p0-cloudy-c23p01"
    )

    # The file landed under its alias, with no partial file left behind
    saved = tmp_path / "test_grid.hdf5"
    assert saved.read_bytes() == PAYLOAD
    assert not (tmp_path / "test_grid.hdf5.part").exists()


def test_a_named_dataset_needs_no_database_entry(
    tmp_path, fake_requests, capsys
):
    """A catalogue name can be downloaded without a local alias."""
    _, calls = fake_requests

    downloader.download_dataset("some-new-grid", str(tmp_path))

    assert calls[0].endswith("/v1/datasets/some-new-grid")
    assert (tmp_path / "test_grid.hdf5").read_bytes() == PAYLOAD
    assert str(tmp_path / "test_grid.hdf5") in capsys.readouterr().out


def test_cli_downloads_each_named_dataset(tmp_path, monkeypatch):
    """--dataset accepts a space-separated list of catalogue names."""
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthesizer-download",
            "--dataset",
            "grid-one",
            "grid-two",
            "--test-grids",
            "--destination",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        downloader,
        "download_dataset",
        lambda dataset, destination, release_id: calls.append(
            (dataset, destination, release_id)
        ),
    )
    monkeypatch.setattr(
        downloader,
        "download_test_grids",
        lambda destination: pytest.fail("dataset branch did not return"),
    )

    downloader.download()

    assert calls == [
        ("grid-one", str(tmp_path), None),
        ("grid-two", str(tmp_path), None),
    ]


def test_cli_cannot_pin_one_release_for_multiple_datasets(monkeypatch):
    """One release id cannot describe several named datasets."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthesizer-download",
            "--dataset",
            "grid-one",
            "grid-two",
            "--release",
            "2",
        ],
    )

    with pytest.raises(exceptions.InconsistentArguments, match="only.*one"):
        downloader.download()


def test_a_named_dataset_is_still_verified(tmp_path, fake_requests):
    """A dataset fetched by name is checked against its published digest."""
    fake_get, _ = fake_requests
    fake_get.catalogue = release(sha256="0" * 64)

    with pytest.raises(exceptions.DownloadError, match="failed verification"):
        downloader.download_dataset("some-new-grid", str(tmp_path))

    assert not (tmp_path / "test_grid.hdf5").exists()


def test_a_named_dataset_can_pin_a_release(tmp_path, monkeypatch):
    """--release asks for one release rather than whatever is current."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "/download" in url:
            return FakeResponse(
                payload=PAYLOAD, headers={"content-length": str(len(PAYLOAD))}
            )
        return FakeResponse(json_data=release()["current_release"])

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    downloader.download_dataset("some-new-grid", str(tmp_path), 2)

    assert calls[0].endswith("/v1/releases/2")
    assert (tmp_path / "test_grid.hdf5").read_bytes() == PAYLOAD


def test_a_pinned_release_is_fetched_instead_of_the_current_one(
    tmp_path, monkeypatch
):
    """An entry can pin a superseded release and still get those bytes."""
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        if "/v1/releases/" in url and not url.endswith("/download"):
            # A pinned release describes itself at the top level.
            return FakeResponse(json_data=release()["current_release"])
        return FakeResponse(
            payload=PAYLOAD, headers={"content-length": str(len(PAYLOAD))}
        )

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setitem(
        downloader.AVAILABLE_FILES,
        "pinned.hdf5",
        {"dataset": "bpass-2-2-1-cloudy-sps", "release": 2},
    )

    downloader._download("pinned.hdf5", str(tmp_path))

    # It asked for the release directly, never for the dataset's current one.
    assert seen[0].endswith("/v1/releases/2")
    assert not any("/v1/datasets/" in call for call in seen)
    assert (tmp_path / "pinned.hdf5").read_bytes() == PAYLOAD


def test_a_database_entry_without_a_dataset_is_an_error(tmp_path, monkeypatch):
    """A stale local entry reports its missing catalogue mapping clearly."""
    monkeypatch.setitem(downloader.AVAILABLE_FILES, "stale.hdf5", {})

    with pytest.raises(exceptions.DownloadError, match="no dataset key"):
        downloader._download("stale.hdf5", str(tmp_path))


def test_an_html_error_page_is_refused(tmp_path, monkeypatch):
    """A stale link answering with HTML must not be saved as a grid."""
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda url, **kwargs: FakeResponse(
            payload=b"<!DOCTYPE html><html>Not found</html>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        ),
    )

    with pytest.raises(exceptions.DownloadError, match="HTML page"):
        downloader._fetch(
            "grid.hdf5", downloader.DATA_API_URL, None, 0, str(tmp_path)
        )

    assert os.listdir(tmp_path) == []


def test_a_mislabelled_error_page_is_refused(tmp_path, monkeypatch):
    """Content-Type cannot be trusted, so the bytes are checked too."""
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda url, **kwargs: FakeResponse(
            payload=b"<!DOCTYPE html><html>Not found</html>",
            headers={"Content-Type": "application/octet-stream"},
        ),
    )

    with pytest.raises(exceptions.DownloadError, match="did not return"):
        downloader._fetch(
            "grid.hdf5", downloader.DATA_API_URL, None, 0, str(tmp_path)
        )

    assert os.listdir(tmp_path) == []


def test_a_format_without_a_signature_is_left_alone(tmp_path, monkeypatch):
    """Formats with no reliable signature are not second-guessed."""
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda url, **kwargs: FakeResponse(payload=b"arbitrary pickle bytes"),
    )

    name = "model.pkl"
    downloader._fetch(name, downloader.DATA_API_URL, None, 0, str(tmp_path))

    assert (tmp_path / name).read_bytes() == b"arbitrary pickle bytes"


def test_digest_mismatch_discards_the_download(tmp_path, fake_requests):
    """Bytes that do not match the published digest are never installed."""
    fake_get, _ = fake_requests
    fake_get.catalogue = release(sha256="0" * 64)

    with pytest.raises(exceptions.DownloadError, match="failed verification"):
        downloader._download("test_grid.hdf5", str(tmp_path))

    # Neither the final file nor the partial download survives
    assert os.listdir(tmp_path) == []


def test_concurrent_cleanup_does_not_hide_verification_error(
    tmp_path, fake_requests, monkeypatch
):
    """Another process removing a partial must not replace DownloadError."""
    fake_get, _ = fake_requests
    fake_get.catalogue = release(sha256="0" * 64)

    def already_removed(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(downloader.os, "remove", already_removed)

    with pytest.raises(exceptions.DownloadError, match="failed verification"):
        downloader._download("test_grid.hdf5", str(tmp_path))


def test_success_discards_partials_from_older_releases(
    tmp_path, fake_requests
):
    """Installing a release removes digest partials it supersedes."""
    stale = tmp_path / "test_grid.hdf5.deadbeefdead.part"
    stale.write_bytes(b"old release")

    downloader._download("test_grid.hdf5", str(tmp_path))

    assert not stale.exists()


def test_truncated_unverifiable_download_is_not_installed(
    tmp_path, monkeypatch
):
    """Content-Length protects downloads that have no published digest."""
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda url, **kwargs: FakeResponse(
            payload=PAYLOAD[:8],
            headers={"Content-Length": str(len(PAYLOAD))},
        ),
    )

    with pytest.raises(exceptions.DownloadError, match="incomplete"):
        downloader._fetch(
            "grid.hdf5", downloader.DATA_API_URL, None, 0, str(tmp_path)
        )

    assert os.listdir(tmp_path) == []


def test_a_dataset_with_no_release_is_an_error(tmp_path, fake_requests):
    """A catalogue entry with nothing published cannot be downloaded."""
    fake_get, _ = fake_requests
    fake_get.catalogue = {"current_release": None}

    with pytest.raises(exceptions.DownloadError, match="no current release"):
        downloader._download("test_grid.hdf5", str(tmp_path))


def test_server_failures_try_each_catalogue_host(tmp_path, monkeypatch):
    """A server failure is retried against each catalogue host."""
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda url, **kwargs: FakeResponse(status_code=503),
    )

    with pytest.raises(exceptions.DownloadError, match="503"):
        downloader._download("test_grid.hdf5", str(tmp_path))


def test_an_unknown_dataset_does_not_try_the_fallback(tmp_path, monkeypatch):
    """A 404 describes the request, so another hostname adds no value."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(status_code=404)

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(exceptions.DownloadError, match="404"):
        downloader.download_dataset("mistyped-dataset", str(tmp_path))

    assert calls == [f"{downloader.DATA_API_URL}/v1/datasets/mistyped-dataset"]


def test_a_blocked_primary_host_falls_back(tmp_path, monkeypatch):
    """A host that cannot be reached is skipped, with a warning."""
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        if url.startswith(downloader.DATA_API_URL):
            # What a newly registered domain looks like behind a filter
            raise downloader.requests.exceptions.SSLError("self-signed cert")
        if "/v1/datasets/" in url:
            return FakeResponse(
                json_data=release(base=downloader.DATA_API_FALLBACK_URL)
            )
        return FakeResponse(
            payload=PAYLOAD, headers={"content-length": str(len(PAYLOAD))}
        )

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    # The warning text is wrapped, so match a phrase that cannot be split
    with pytest.warns(RuntimeWarning, match="was unavailable"):
        downloader._download("test_grid.hdf5", str(tmp_path))

    # The primary was tried first, then the fallback served the metadata
    assert seen[0].startswith(downloader.DATA_API_URL)
    assert seen[1].startswith(downloader.DATA_API_FALLBACK_URL)

    # The bytes came from the host that answered, not the blocked one
    assert seen[2].startswith(downloader.DATA_API_FALLBACK_URL)
    assert (tmp_path / "test_grid.hdf5").read_bytes() == PAYLOAD


@pytest.mark.parametrize("bad_payload", [ValueError("bad JSON"), []])
def test_an_invalid_catalogue_response_does_not_fall_back(
    tmp_path, monkeypatch, bad_payload
):
    """Malformed metadata is reported without trying another hostname."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(json_data=bad_payload)

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(exceptions.DownloadError):
        downloader._download("test_grid.hdf5", str(tmp_path))

    assert len(calls) == 1


def test_incomplete_release_metadata_does_not_fall_back(tmp_path, monkeypatch):
    """Incomplete metadata is not repaired by changing hostname."""
    incomplete = release()
    del incomplete["current_release"]["file"]["sha256"]
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(json_data=incomplete)

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(exceptions.DownloadError, match="Invalid release"):
        downloader.download_dataset("some-new-grid", str(tmp_path))

    assert len(calls) == 1


def test_catalogue_filename_cannot_escape_destination(tmp_path, monkeypatch):
    """Catalogue filenames must be safe basenames before fetching."""
    malicious = release()
    malicious["current_release"]["file"]["filename"] = "../escaped.hdf5"
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(json_data=malicious)

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    destination = tmp_path / "downloads"

    with pytest.raises(exceptions.DownloadError, match="Invalid release"):
        downloader.download_dataset("some-new-grid", str(destination))

    assert all("/v1/datasets/" in url for url in calls)
    assert not (tmp_path / "escaped.hdf5").exists()


def test_all_hosts_failing_reports_every_reason(tmp_path, monkeypatch):
    """When no host works, each host's reason is surfaced."""

    def fake_get(url, **kwargs):
        if url.startswith(downloader.DATA_API_URL):
            raise downloader.requests.exceptions.SSLError("self-signed cert")
        return FakeResponse(status_code=503)

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(exceptions.DownloadError) as excinfo:
        downloader._download("test_grid.hdf5", str(tmp_path))

    message = str(excinfo.value)
    assert downloader.DATA_API_URL in message
    assert downloader.DATA_API_FALLBACK_URL in message
    assert "503" in message


def test_a_tls_failure_explains_certificate_interception(
    tmp_path, monkeypatch
):
    """TLS errors name the likely cause, not urllib3 internals."""

    def fake_get(url, **kwargs):
        raise downloader.requests.exceptions.SSLError(
            "self-signed certificate"
        )

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(exceptions.DownloadError, match="SSL_CERT_FILE"):
        downloader._download("test_grid.hdf5", str(tmp_path))


class DyingResponse(FakeResponse):
    """A response that dies part way through the body."""

    def iter_content(self, block_size):
        """Yield a few bytes and then fail.

        Args:
            block_size (int): The number of bytes per chunk.
        """
        yield PAYLOAD[:4]
        raise OSError("connection reset")


def test_interrupted_download_keeps_a_resumable_partial(tmp_path, monkeypatch):
    """A verifiable transfer leaves its partial file for the next attempt."""

    def fake_get(url, **kwargs):
        if "/v1/datasets/" in url:
            return FakeResponse(json_data=release())
        return DyingResponse(headers={"content-length": str(len(PAYLOAD))})

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(OSError, match="connection reset"):
        downloader._download("test_grid.hdf5", str(tmp_path))

    # The finished file was never created, but the partial survives, named
    # after the digest so it can only be resumed into the same release.
    assert not (tmp_path / "test_grid.hdf5").exists()
    partials = list(tmp_path.glob("*.part"))
    assert len(partials) == 1
    assert DIGEST[:12] in partials[0].name
    assert partials[0].read_bytes() == PAYLOAD[:4]


def test_an_unverifiable_interrupted_download_is_discarded(
    tmp_path, monkeypatch
):
    """Without a digest a partial file cannot be trusted, so it goes."""
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda url, **kwargs: DyingResponse(
            headers={"content-length": str(len(PAYLOAD))}
        ),
    )

    with pytest.raises(OSError, match="connection reset"):
        downloader._fetch(
            "grid.hdf5", downloader.DATA_API_URL, None, 0, str(tmp_path)
        )

    assert os.listdir(tmp_path) == []


def test_a_partial_download_resumes_from_where_it_stopped(
    tmp_path, monkeypatch
):
    """A second attempt asks only for the bytes it is missing."""
    part = tmp_path / f"test_grid.hdf5.{DIGEST[:12]}.part"
    part.write_bytes(PAYLOAD[:4])
    ranges = []

    def fake_get(url, **kwargs):
        if "/v1/datasets/" in url:
            return FakeResponse(json_data=release())
        ranges.append((kwargs.get("headers") or {}).get("Range"))
        return FakeResponse(
            status_code=206,
            payload=PAYLOAD[4:],
            headers={
                "content-length": str(len(PAYLOAD) - 4),
                "Content-Range": f"bytes 4-{len(PAYLOAD) - 1}/{len(PAYLOAD)}",
            },
        )

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    downloader._download("test_grid.hdf5", str(tmp_path))

    # Only the missing tail was requested, and the whole file verifies
    assert ranges == ["bytes=4-"]
    assert (tmp_path / "test_grid.hdf5").read_bytes() == PAYLOAD
    assert list(tmp_path.glob("*.part")) == []


def test_a_server_ignoring_the_range_restarts_cleanly(tmp_path, monkeypatch):
    """A 200 answer to a range request replaces the partial file."""
    part = tmp_path / f"test_grid.hdf5.{DIGEST[:12]}.part"
    part.write_bytes(b"stale nonsense")

    def fake_get(url, **kwargs):
        if "/v1/datasets/" in url:
            return FakeResponse(json_data=release())
        # Ignoring Range and sending the whole body is legal
        return FakeResponse(
            payload=PAYLOAD, headers={"content-length": str(len(PAYLOAD))}
        )

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    downloader._download("test_grid.hdf5", str(tmp_path))

    # The stale bytes were overwritten rather than appended to
    assert (tmp_path / "test_grid.hdf5").read_bytes() == PAYLOAD


def test_an_unrequested_partial_response_is_rejected(tmp_path, monkeypatch):
    """A 206 response is valid only for a validated resume request."""
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda url, **kwargs: FakeResponse(status_code=206, payload=PAYLOAD),
    )
    with pytest.raises(exceptions.DownloadError, match="206"):
        downloader._fetch(
            "grid.hdf5", downloader.DATA_API_URL, None, 0, str(tmp_path)
        )


@pytest.mark.parametrize(
    "partial_headers",
    [
        # A 206 for a range we did not ask for cannot be appended
        {"Content-Range": "bytes 0-14/15"},
        # Nor can one that does not say which bytes it holds
        {},
    ],
)
def test_a_mismatched_partial_response_restarts(
    tmp_path, monkeypatch, partial_headers
):
    """Only a continuation from our offset may be appended."""
    part = tmp_path / f"test_grid.hdf5.{DIGEST[:12]}.part"
    part.write_bytes(PAYLOAD[:4])
    calls = []

    def fake_get(url, **kwargs):
        if "/v1/datasets/" in url:
            return FakeResponse(json_data=release())
        calls.append((kwargs.get("headers") or {}).get("Range"))
        if len(calls) == 1:
            # An unusable partial response to the resume attempt
            return FakeResponse(
                status_code=206,
                payload=PAYLOAD,
                headers={
                    "content-length": str(len(PAYLOAD)),
                    **partial_headers,
                },
            )
        return FakeResponse(
            payload=PAYLOAD, headers={"content-length": str(len(PAYLOAD))}
        )

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    downloader._download("test_grid.hdf5", str(tmp_path))

    # It asked to resume, then restarted without a Range header
    assert calls[0] == "bytes=4-"
    assert calls[1] is None
    assert (tmp_path / "test_grid.hdf5").read_bytes() == PAYLOAD


def test_a_transport_failure_becomes_a_download_error(tmp_path, monkeypatch):
    """Transport errors surface as DownloadError, not raw requests errors."""

    class DyingConnection(FakeResponse):
        def iter_content(self, block_size):
            yield PAYLOAD[:4]
            raise downloader.requests.exceptions.ChunkedEncodingError(
                "connection dropped"
            )

    def fake_get(url, **kwargs):
        if "/v1/datasets/" in url:
            return FakeResponse(json_data=release())
        return DyingConnection(headers={"content-length": str(len(PAYLOAD))})

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(exceptions.DownloadError, match="will resume it"):
        downloader._download("test_grid.hdf5", str(tmp_path))

    # The partial is kept precisely so that resuming is possible
    assert len(list(tmp_path.glob("*.part"))) == 1


def test_a_corrupt_partial_is_removed_so_a_retry_is_clean(
    tmp_path, monkeypatch
):
    """A resumed file that fails verification does not poison the next try."""
    part = tmp_path / f"test_grid.hdf5.{DIGEST[:12]}.part"
    part.write_bytes(b"XXXX")

    def fake_get(url, **kwargs):
        if "/v1/datasets/" in url:
            return FakeResponse(json_data=release())
        return FakeResponse(
            status_code=206,
            payload=PAYLOAD[4:],
            headers={
                "content-length": str(len(PAYLOAD) - 4),
                "Content-Range": f"bytes 4-{len(PAYLOAD) - 1}/{len(PAYLOAD)}",
            },
        )

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(exceptions.DownloadError, match="failed verification"):
        downloader._download("test_grid.hdf5", str(tmp_path))

    assert os.listdir(tmp_path) == []


def test_every_entry_names_a_dataset():
    """Every downloadable file resolves through the catalogue."""
    for entries in downloader.load_database_yaml().values():
        for name, entry in entries.items():
            assert "dataset" in entry, name
            assert set(entry) <= {"dataset", "release"}, name

    # Every alias target must be one of them, or aliases would break
    for alias, target in downloader.TEST_DATA_TRANSLATION.items():
        assert "dataset" in downloader.AVAILABLE_FILES[target], alias
