import json
from pathlib import Path

import pytest

from eo_visual_retrieval.models import StacItemRecord
from eo_visual_retrieval.stac import StacSearchConfig, read_stac_jsonl, write_stac_jsonl


def test_stac_config_rejects_unbounded_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        StacSearchConfig(
            api_url="https://example.test/stac",
            collection="example",
            bbox=(1.0, 2.0, 3.0, 4.0),
            datetime="2024-01-01/2024-01-31",
            limit=1001,
        )


def test_stac_manifest_contains_identity_but_no_hrefs(tmp_path: Path) -> None:
    record = StacItemRecord(
        api_url="https://example.test/stac",
        collection="sentinel",
        item_id="item-1",
        bbox=(1.0, 2.0, 3.0, 4.0),
        datetime="2024-01-01T00:00:00+00:00",
        asset_keys=("blue", "green", "red"),
        properties={"eo:cloud_cover": 5.0},
    )
    output = tmp_path / "stac.jsonl"
    write_stac_jsonl([record], output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["item_id"] == "item-1"
    assert "href" not in output.read_text(encoding="utf-8").lower()
    assert read_stac_jsonl(output) == [record]
