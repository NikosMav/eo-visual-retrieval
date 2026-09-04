import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from eo_visual_retrieval.chips import (
    build_sentinel2_chip,
    sentinel2_reflectance_parameters,
)


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    transform: object,
    nodata: int = 0,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32610",
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(values, 1)


def _sources(tmp_path: Path) -> dict[str, Path]:
    transform_10m = from_origin(500000, 1000, 10, 10)
    for key, value in (("B04", 2000), ("B03", 3000), ("B02", 4000)):
        _write_raster(
            tmp_path / f"{key}.tif",
            np.full((4, 4), value, dtype=np.uint16),
            transform=transform_10m,
        )
    scl = np.asarray([[4, 9], [4, 4]], dtype=np.uint8)
    _write_raster(
        tmp_path / "SCL.tif",
        scl,
        transform=from_origin(500000, 1000, 20, 20),
    )
    return {key: tmp_path / f"{key}.tif" for key in ("B04", "B03", "B02", "SCL")}


def test_processing_baseline_controls_reflectance_offset() -> None:
    assert sentinel2_reflectance_parameters("03.01") == (0.0001, 0.0)
    assert sentinel2_reflectance_parameters("04.00") == (0.0001, -0.1)
    assert sentinel2_reflectance_parameters("05.10") == (0.0001, -0.1)
    with pytest.raises(ValueError, match="processing baseline"):
        sentinel2_reflectance_parameters("unknown")


def test_chip_aligns_scales_masks_and_records_metadata(tmp_path: Path) -> None:
    artifacts = build_sentinel2_chip(
        _sources(tmp_path),
        output_dir=tmp_path / "chips",
        item_id="sentinel/item-1",
        api_url="https://example.test/stac",
        collection="sentinel-2-l2a",
        datetime="2024-06-25T00:00:00Z",
        bounds=(500000, 960, 500040, 1000),
        bounds_crs="EPSG:32610",
        processing_baseline="05.10",
    )

    assert artifacts.reflectance_path.is_file()
    assert artifacts.rgb_path.is_file()
    assert artifacts.image_record.path == "sentinel-item-1-rgb.tif"
    assert artifacts.image_record.source == "stac-sentinel2-chip"
    assert artifacts.image_record.metadata["width"] == 4
    assert artifacts.image_record.metadata["height"] == 4
    assert artifacts.image_record.metadata["gsd"] == [10.0, 10.0]
    serialized = json.dumps(artifacts.image_record.to_dict()).lower()
    assert "href" not in serialized
    assert "token" not in serialized
    assert "signature" not in serialized

    with rasterio.open(artifacts.reflectance_path) as dataset:
        reflectance = dataset.read()
        mask = dataset.dataset_mask()
        assert dataset.crs.to_string() == "EPSG:32610"
        assert dataset.transform == from_origin(500000, 1000, 10, 10)
        np.testing.assert_allclose(reflectance[:, 3, 0], [0.1, 0.2, 0.3], atol=1e-7)
        assert mask[0, 3] == 0
        assert mask[3, 0] == 255

    with rasterio.open(artifacts.rgb_path) as dataset:
        rgb = dataset.read()
        np.testing.assert_array_equal(rgb[:, 3, 0], [85, 170, 255])
        np.testing.assert_array_equal(rgb[:, 0, 3], [0, 0, 0])
        assert dataset.dataset_mask()[0, 3] == 0

    with Image.open(artifacts.rgb_path) as image:
        assert image.mode == "RGB"
        assert image.size == (4, 4)


def test_chip_enforces_pixel_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit is 15"):
        build_sentinel2_chip(
            _sources(tmp_path),
            output_dir=tmp_path / "chips",
            item_id="item-1",
            api_url="https://example.test/stac",
            collection="sentinel-2-l2a",
            datetime=None,
            bounds=(500000, 960, 500040, 1000),
            bounds_crs="EPSG:32610",
            processing_baseline="05.10",
            max_pixels=15,
        )


def test_chip_rejects_bounds_outside_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not overlap"):
        build_sentinel2_chip(
            _sources(tmp_path),
            output_dir=tmp_path / "chips",
            item_id="item-1",
            api_url="https://example.test/stac",
            collection="sentinel-2-l2a",
            datetime=None,
            bounds=(600000, 960, 600040, 1000),
            bounds_crs="EPSG:32610",
            processing_baseline="05.10",
        )


def test_chip_rejects_access_bearing_provenance_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "chips"
    with pytest.raises(ValueError, match="api_url"):
        build_sentinel2_chip(
            {}, output_dir=output, item_id="test",
            api_url="https://example.test/stac?api_key=synthetic-value",
            collection="sentinel-2-l2a", datetime=None,
            bounds=(500000, 960, 500040, 1000), bounds_crs="EPSG:32610",
            processing_baseline="05.10",
        )
    assert not output.exists()
