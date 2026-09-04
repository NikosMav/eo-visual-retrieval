"""Dataset identity, band mapping, and archive access shared by every encoder."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from eo_visual_retrieval.datasets.eurosat import (
    EUROSAT_ARCHIVE_MD5,
    EUROSAT_BAND_ORDER,
    EUROSAT_CLASSES,
    archive_members,
    band_indices,
    read_archive_member,
    verify_archive,
)
from eo_visual_retrieval.hashing import file_md5, file_sha256, verify_sha256
from eo_visual_retrieval.models import ImageRecord
from eo_visual_retrieval.vectors import l2_normalize


def _record(item_id: str, member: str | None, source: str = "eurosat-ms-v1") -> ImageRecord:
    metadata = {} if member is None else {"archive_member": member}
    return ImageRecord(
        item_id=item_id, path=item_id, split="index", source=source, metadata=metadata
    )


def test_band_indices_place_b8a_where_sentinel2_models_expect_it() -> None:
    """EuroSAT stores B8A last; models expect it between B08 and B09."""

    assert EUROSAT_BAND_ORDER[-1] == "B8A"
    ordered = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09")

    indices = band_indices(ordered)

    assert indices[8] == EUROSAT_BAND_ORDER.index("B8A") == 12
    assert indices[9] == EUROSAT_BAND_ORDER.index("B09") == 8


def test_band_indices_reject_unknown_or_repeated_bands() -> None:
    with pytest.raises(ValueError, match="unknown EuroSAT band: B99"):
        band_indices(("B01", "B99"))
    with pytest.raises(ValueError, match="must not repeat a band"):
        band_indices(("B01", "B01"))


def test_archive_members_reject_foreign_incomplete_and_duplicate_records() -> None:
    assert archive_members([_record("a", "m/a.tif"), _record("b", "m/b.tif")]) == [
        "m/a.tif",
        "m/b.tif",
    ]

    with pytest.raises(ValueError, match="at least one image record"):
        archive_members([])
    with pytest.raises(ValueError, match="is not a eurosat-ms-v1 item"):
        archive_members([_record("a", "m/a.tif", source="local")])
    with pytest.raises(ValueError, match="lacks an archive member"):
        archive_members([_record("a", None)])
    with pytest.raises(ValueError, match="duplicate EuroSAT archive members"):
        archive_members([_record("a", "m/a.tif"), _record("b", "m/a.tif")])


def test_verify_archive_checks_existence_and_published_checksum(tmp_path: Path) -> None:
    archive = tmp_path / "EuroSAT_MS.zip"

    with pytest.raises(ValueError, match="archive does not exist"):
        verify_archive(archive)

    archive.write_bytes(b"not the published archive")
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        verify_archive(archive, expected_md5=EUROSAT_ARCHIVE_MD5)

    # An explicit opt-out is how discovery reads a locally prepared archive.
    verify_archive(archive, expected_md5=None)
    verify_archive(archive, expected_md5=file_md5(archive))


def test_read_archive_member_requires_thirteen_bands(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    archive = tmp_path / "patches.zip"
    profile = {
        "driver": "GTiff",
        "width": 4,
        "height": 4,
        "crs": "EPSG:32633",
        "transform": from_origin(0, 0, 10, 10),
        "dtype": "uint16",
    }

    def raster(bands: int) -> bytes:
        with rasterio.io.MemoryFile() as memory:
            with memory.open(count=bands, **profile) as dataset:
                for band in range(1, bands + 1):
                    dataset.write(np.full((4, 4), band * 100, dtype=np.uint16), band)
            return memory.read()

    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ok.tif", raster(13))
        bundle.writestr("too-few.tif", raster(4))

    with zipfile.ZipFile(archive) as bundle:
        values = read_archive_member(bundle, "ok.tif")
        assert values.shape == (13, 4, 4)
        assert values[12, 0, 0] == 1300

        with pytest.raises(ValueError, match="does not have 13 bands"):
            read_archive_member(bundle, "too-few.tif")
        with pytest.raises(ValueError, match="member does not exist"):
            read_archive_member(bundle, "absent.tif")


def test_class_list_matches_the_published_dataset() -> None:
    assert len(EUROSAT_CLASSES) == 10
    assert EUROSAT_CLASSES == tuple(sorted(EUROSAT_CLASSES))


def test_digests_are_streamed_and_verified(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(b"checkpoint")
    expected = "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"

    assert file_sha256(path) == expected
    assert verify_sha256(path, expected.upper()) == expected
    assert len(file_md5(path)) == 32

    with pytest.raises(ValueError, match="checkpoint does not exist"):
        verify_sha256(tmp_path / "absent.pth", expected)


def test_normalization_rejects_input_that_cannot_be_ranked() -> None:
    np.testing.assert_allclose(
        l2_normalize(np.asarray([[3.0, 4.0]], dtype=np.float32)), [[0.6, 0.8]], rtol=1e-6
    )

    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        l2_normalize(np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        l2_normalize(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="only finite values"):
        l2_normalize(np.asarray([[np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="zero-length rows"):
        l2_normalize(np.zeros((1, 3), dtype=np.float32))


def test_normalization_rejects_inputs_whose_float32_norm_overflows() -> None:
    """Finite inputs can still square to infinity, which silently zeroed the row.

    ``matrix / inf`` returns zeros, so the zero-length guard above cannot catch
    this: it runs before the division. An unranked zero vector must be an error
    rather than a row that quietly stops matching anything.
    """

    overflowing = np.asarray([[1e20, 1e20]], dtype=np.float32)
    assert np.isfinite(overflowing).all()

    with pytest.raises(ValueError, match="norm overflowed float32"):
        l2_normalize(overflowing)
