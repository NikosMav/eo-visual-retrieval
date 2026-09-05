"""Local multimodal search HTTP surface; inference and ranking live outside HTTP."""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from eo_visual_retrieval.app.catalog import Catalog
from eo_visual_retrieval.app.evidence import Evidence
from eo_visual_retrieval.app.main import ContentLengthLimitMiddleware, _headline, create_app
from eo_visual_retrieval.app.thumbnails import thumbnail_jpeg
from eo_visual_retrieval.app.uploads import MAX_UPLOAD_BYTES, MAX_UPLOAD_PIXELS
from eo_visual_retrieval.multimodal import MultimodalSearch
from eo_visual_retrieval.search_plan import SearchFilters, SearchPlan, plan_query


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    text: str = Field(default="", max_length=1000)
    item_id: str | None = Field(default=None, max_length=500)
    text_weight: float = Field(default=0.5, ge=0, le=1)
    k: int = Field(default=12, ge=1, le=100)
    bbox: tuple[float, float, float, float] | None = None
    start_date: date | None = None
    end_date: date | None = None
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    collection: str | None = Field(default=None, max_length=100)
    interpret: bool = True

    def plan(self) -> SearchPlan:
        return plan_query(
            self.text,
            interpret=self.interpret,
            overrides=SearchFilters(
                bbox=self.bbox,
                start_date=self.start_date,
                end_date=self.end_date,
                max_cloud_cover=self.max_cloud_cover,
                collection=self.collection,
            ),
        )


def decode_search_image(data: bytes) -> Image.Image:
    """Validate bounded RGB upload without changing the encoder's crop/resize."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("image must be nonempty and no larger than 8 MiB")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_UPLOAD_PIXELS:
                raise ValueError("image exceeds the 16-megapixel limit")
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            return source.convert("RGB")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("upload is not a readable RGB image") from error


def create_search_app(
    engine: MultimodalSearch, *, results_dir: Path | None = None,
    comparison: Catalog | None = None,
) -> FastAPI:
    app = FastAPI(title="EO text, image and hybrid search")
    root = Path(__file__).parent
    templates = Jinja2Templates(directory=str(root / "templates"))
    app.mount("/static", StaticFiles(directory=root / "static"), name="static")
    app.add_middleware(ContentLengthLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)
    evidence = Evidence(results_dir)
    if comparison is not None:
        app.mount("/models", create_app(comparison, k=min(5, comparison.index_size),
                                       product_navigation=True))

    def render(request: Request, name: str, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name=name, context={
            "corpus": engine.describe(), "comparison_available": comparison is not None,
            "evidence": evidence.payload(), **context,
        })

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {"error": "Invalid search fields; check dates, bounds and weights."}, 422
        )

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return render(request, "search.html", active="search")

    if comparison is None:
        @app.get("/models/", response_class=HTMLResponse)
        def models_unavailable(request: Request) -> HTMLResponse:
            return render(request, "research.html", active="models", missing_models=True)

    @app.get("/findings", response_class=HTMLResponse)
    def findings(request: Request) -> HTMLResponse:
        return render(request, "evidence.html", active="findings")

    @app.get("/research", response_class=HTMLResponse)
    def research(request: Request) -> HTMLResponse:
        return render(request, "research.html", active="research")

    @app.get("/api/evidence")
    def evidence_api() -> dict[str, Any]:
        return evidence.payload()

    @app.get("/evidence/{name}")
    def evidence_file(name: str) -> Response:
        raw = evidence.files.get(name)
        if raw is None:
            return Response("evidence unavailable", status_code=404)
        return Response(raw, media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    if evidence.analysis is not None:
        @app.get("/findings/analysis", response_class=HTMLResponse)
        def detailed_findings(request: Request) -> HTMLResponse:
            assert evidence.analysis is not None
            return render(request, "findings.html", active="findings",
                          product_navigation=True, findings=evidence.analysis,
                          headline=_headline(evidence.analysis),
                          findings_json=json.dumps(evidence.analysis).replace("<", "\\u003c"))

    @app.get("/api/corpus")
    def corpus() -> dict[str, Any]:
        return engine.describe()

    @app.post("/api/plan")
    def plan(payload: SearchInput) -> Any:
        try:
            return payload.plan().to_dict()
        except ValueError as error:
            return JSONResponse({"error": str(error)}, 400)

    @app.post("/api/search")
    async def search(request: Request) -> Any:
        image = None
        try:
            async with request.form(max_files=1, max_fields=1) as form:
                raw = form.get("query")
                if not isinstance(raw, str) or len(raw) > 5000:
                    raise ValueError("provide a bounded JSON query field")
                payload = SearchInput.model_validate_json(raw)
                plan = payload.plan()
                upload = form.get("image")
                if upload is not None:
                    if not isinstance(upload, UploadFile):
                        raise ValueError("image must be an uploaded file")
                    data = await upload.read(MAX_UPLOAD_BYTES + 1)
                    image = await run_in_threadpool(decode_search_image, data)
                return await run_in_threadpool(
                    engine.search,
                    plan,
                    image=image,
                    item_id=payload.item_id,
                    text_weight=payload.text_weight,
                    k=payload.k,
                )
        except ValidationError:
            return JSONResponse(
                {"error": "Invalid search fields; check dates, bounds and weights."},
                status_code=422,
            )
        except ValueError as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        finally:
            if image is not None:
                image.close()

    @app.get("/thumbnail")
    def thumbnail(item_id: str) -> Response:
        try:
            return Response(
                thumbnail_jpeg(engine.image_path(item_id), size=256), media_type="image/jpeg"
            )
        except (ValueError, OSError):
            return Response("thumbnail unavailable", status_code=404)

    return app
