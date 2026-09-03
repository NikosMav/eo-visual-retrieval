"""HTTP routing for the comparison surface.

This is the only module here that knows about HTTP. All ranking decisions live
in catalog.py so they can be tested without a server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from eo_visual_retrieval.app.catalog import Catalog
from eo_visual_retrieval.app.thumbnails import thumbnail_jpeg
from eo_visual_retrieval.app.uploads import MAX_UPLOAD_BYTES, decode_upload

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
THUMBNAIL_PIXELS = 96


def create_app(catalog: Catalog, *, k: int = 5) -> FastAPI:
    """Build the application around one already-validated catalog."""

    if k < 1:
        raise ValueError("k must be positive")
    app = FastAPI(title="EO visual retrieval")

    def render(
        request: Request,
        rankings: list[Any],
        *,
        query_id: str | None,
        is_upload: bool,
    ) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="compare.html",
            context={
                "rankings": rankings,
                "query_ids": catalog.query_ids,
                "item_id": query_id,
                "query_id": query_id,
                "is_upload": is_upload,
                "upload_available": catalog.upload_available,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        first = catalog.query_ids[0]
        return render(request, catalog.rank_item(first, k=k), query_id=first, is_upload=False)

    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request, item_id: str) -> HTMLResponse:
        try:
            rankings = catalog.rank_item(item_id, k=k)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return render(request, rankings, query_id=item_id, is_upload=False)

    @app.post("/compare/upload", response_class=HTMLResponse)
    async def compare_upload(
        request: Request, image: UploadFile = File(...)  # noqa: B008
    ) -> HTMLResponse:
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        try:
            pixels = decode_upload(data, image_size=catalog.image_size)
            ranking = catalog.rank_uploaded(pixels, k=k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return render(request, [ranking], query_id=None, is_upload=True)

    @app.get("/thumbnail")
    def thumbnail(item_id: str) -> Response:
        try:
            path = catalog.image_path(item_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(
            content=thumbnail_jpeg(path, size=THUMBNAIL_PIXELS), media_type="image/jpeg"
        )

    return app
