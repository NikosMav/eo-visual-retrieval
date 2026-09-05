"""The findings page must read the committed evidence, never restate it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eo_visual_retrieval.app.findings import (
    ANALYSIS_FILE,
    AVAILABILITY_FILE,
    build_payload,
    load_findings,
)

REPO_RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"


def _analysis(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "k": 10,
        "aggregate": {
            "corpus-alpha": {"map_at_k": 0.80},
            "corpus-beta": {"map_at_k": 0.40},
        },
        "agreement": {
            "queries": 200,
            "pairwise_overlap": {"corpus-alpha vs corpus-beta": {"mean_overlap_at_k": 0.25}},
            "top_one_correctness": {
                "all_stores_correct": 50,
                "disputed": 140,
                "exactly_one_store_correct": 10,
                "no_store_correct": 10,
            },
        },
        "geographic": {
            "spatial_group_spread": {
                "corpus-alpha": {
                    "group_mean_map_at_k": {
                        "standard_deviation": 0.10, "minimum": 0.5, "median": 0.8
                    }
                },
                "corpus-beta": {
                    "group_mean_map_at_k": {
                        "standard_deviation": 0.20, "minimum": 0.1, "median": 0.4
                    }
                },
            }
        },
        "distance": {
            "nearest_same_label_index_m": {
                "corpus-alpha": {
                    "quartile_edges_m": [5000.0, 20000.0, 30000.0, 70000.0, 900000.0],
                    "quartiles": {
                        "q1": {"map_at_k": 0.81}, "q2": {"map_at_k": 0.83},
                        "q3": {"map_at_k": 0.79}, "q4": {"map_at_k": 0.77},
                    },
                },
                "corpus-beta": {
                    "quartile_edges_m": [5000.0, 20000.0, 30000.0, 70000.0, 900000.0],
                    "quartiles": {
                        "q1": {"map_at_k": 0.45}, "q2": {"map_at_k": 0.44},
                        "q3": {"map_at_k": 0.39}, "q4": {"map_at_k": 0.30},
                    },
                },
            }
        },
        "failures": {
            "universally_hard_queries": {
                "count": 2,
                "examples": [{"query_id": "a", "label": "River"},
                             {"query_id": "b", "label": "Highway"}],
            }
        },
    }
    payload.update(overrides)
    return payload


def _availability() -> dict[str, Any]:
    return {
        "datetime": "2024-01-01/2024-12-31",
        "max_cloud_cover": 10.0,
        "collection": "sentinel-2-l2a",
        "distinct_days": {"total": 40},
        "results": [
            {"place_id": "po-valley-cremona", "latitude": 45.14, "distinct_days": 30},
            {"place_id": "lapland-rovaniemi", "latitude": 66.5, "distinct_days": 10},
        ],
    }


def _write(directory: Path, analysis: dict[str, Any] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ANALYSIS_FILE).write_text(
        json.dumps(analysis if analysis is not None else _analysis()), encoding="utf-8"
    )
    (directory / AVAILABILITY_FILE).write_text(json.dumps(_availability()), encoding="utf-8")
    return directory


def test_payload_carries_the_recorded_values_unchanged(tmp_path: Path) -> None:
    payload = build_payload(_write(tmp_path / "results"))

    assert payload["k"] == 10
    assert payload["queries"] == 200
    assert [row["key"] for row in payload["series"]] == ["corpus-alpha", "corpus-beta"]
    alpha = payload["series"][0]
    assert alpha["map"] == 0.80
    assert alpha["sd"] == 0.10
    assert alpha["q"] == [0.81, 0.83, 0.79, 0.77]
    assert payload["overlap_max"] == 0.25
    assert payload["quartile_edges_km"] == [5.0, 20.0, 30.0, 70.0, 900.0]
    assert payload["universally_hard"] == {"count": 2, "examples": ["River", "Highway"]}


def test_series_are_ordered_by_quality(tmp_path: Path) -> None:
    """Chart colour follows the entity, so the order must be stable and meaningful."""

    payload = build_payload(_write(tmp_path / "results"))
    scores = [row["map"] for row in payload["series"]]

    assert scores == sorted(scores, reverse=True)


def test_places_run_south_to_north_with_readable_labels(tmp_path: Path) -> None:
    payload = build_payload(_write(tmp_path / "results"))

    assert [place["latitude"] for place in payload["places"]] == [45.14, 66.5]
    # The trailing local specifier is dropped; the full identifier is retained.
    assert payload["places"][0]["label"] == "Po valley"
    assert payload["places"][0]["place_id"] == "po-valley-cremona"
    assert payload["places"][1]["label"] == "Lapland"


def test_missing_evidence_is_reported_not_papered_over(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="cannot read evidence file"):
        build_payload(empty)
    with pytest.raises(ValueError, match="results directory does not exist"):
        load_findings(tmp_path / "absent")
    assert load_findings(None) is None


def test_a_report_missing_a_section_is_refused(tmp_path: Path) -> None:
    """A findings page with holes in it is worse than no findings page."""

    incomplete = _analysis()
    del incomplete["failures"]
    directory = _write(tmp_path / "partial", analysis=incomplete)

    with pytest.raises(ValueError, match="no 'failures' section"):
        build_payload(directory)


def test_committed_reports_still_satisfy_the_page() -> None:
    """The published evidence must keep rendering; a schema change breaks here first."""

    payload = build_payload(REPO_RESULTS)

    assert len(payload["series"]) == 5
    assert len(payload["overlap"]) == 10
    assert len(payload["places"]) == 12
    assert payload["queries"] == 400
    # Values match the published reports exactly, not a copy kept in the page.
    published = json.loads((REPO_RESULTS / ANALYSIS_FILE).read_text(encoding="utf-8"))
    best = payload["series"][0]
    assert best["map"] == max(
        row["map_at_k"] for row in published["aggregate"].values()
    )


def test_embedded_payload_survives_template_escaping(tmp_path: Path) -> None:
    """Autoescaping turned the payload into HTML entities and broke every chart.

    The data reaches the browser inside a script block, so it must arrive as
    valid JSON, with only the one character that can close that block escaped.
    """

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from eo_visual_retrieval.app.main import create_app

    payload = build_payload(_write(tmp_path / "results"))

    class _Catalog:
        query_ids = ("a",)
        index_size = 4
        upload_available = False

        def label(self, item_id: str) -> str:
            return "forest"

        def rank_item(self, item_id: str, *, k: int) -> list[Any]:
            return []

    client = TestClient(create_app(_Catalog(), k=1, findings=payload))  # type: ignore[arg-type]
    body = client.get("/findings").text
    start = body.index('id="findings-data"')
    embedded = body[body.index(">", start) + 1 : body.index("</script>", start)]

    assert "&#34;" not in embedded and "&quot;" not in embedded
    assert "<" not in embedded, "an unescaped < could close the script block early"
    assert json.loads(embedded)["series"][0]["key"] == "corpus-alpha"
