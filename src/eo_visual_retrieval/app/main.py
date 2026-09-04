"""HTTP routing for the comparison surface.

This is the only module here that knows about HTTP. All ranking decisions live
in catalog.py so they can be tested without a server.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from eo_visual_retrieval.app.catalog import Catalog, ModelRanking
from eo_visual_retrieval.app.thumbnails import thumbnail_jpeg
from eo_visual_retrieval.app.uploads import MAX_UPLOAD_BYTES, decode_upload

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
THUMBNAIL_PIXELS = 96


class ContentLengthLimitMiddleware:
    """Reject an oversized request by its declared ``Content-Length`` header.

    This runs ahead of Starlette's multipart form parser, which spools any file
    part larger than 1 MB to an on-disk temporary file and enforces its own size
    limit only on non-file fields (see ``starlette/formparsers.py``). Without this
    guard, a large upload would be fully written to disk before ``decode_upload``
    ever got a chance to refuse it.

    A chunked request has no ``Content-Length`` header and is not covered here;
    a real deployment needs a reverse-proxy body-size limit in front of this
    process for that case.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            declared = headers.get(b"content-length")
            if declared is not None:
                try:
                    length = int(declared)
                except ValueError:
                    length = None
                if length is not None and length > self._max_bytes:
                    response = PlainTextResponse("upload exceeds size limit", status_code=413)
                    await response(scope, receive, send)
                    return
        await self._app(scope, receive, send)


def create_app(catalog: Catalog, *, k: int = 5) -> FastAPI:
    """Build the application around one already-validated catalog."""

    if k < 1:
        raise ValueError("k must be positive")
    if k > catalog.index_size:
        raise ValueError(
            f"k={k} exceeds the index size {catalog.index_size}: lower --k or "
            "supply a larger corpus"
        )
    app = FastAPI(title="EO visual retrieval")
    app.add_middleware(ContentLengthLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)

    def render(
        request: Request,
        rankings: list[ModelRanking],
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
            # decode_upload does real image-library CPU work; running it inline in
            # an async handler would block the event loop and stall every other
            # request for the duration of one decode.
            pixels = await run_in_threadpool(decode_upload, data, image_size=catalog.image_size)
            ranking = catalog.rank_uploaded(pixels, k=k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return render(request, [ranking], query_id=None, is_upload=True)

    @app.get("/thumbnail")
    def thumbnail(item_id: str) -> Response:
        try:
            path = catalog.image_path(item_id)
            content = thumbnail_jpeg(path, size=THUMBNAIL_PIXELS)
        except (KeyError, ValueError, OSError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(content=content, media_type="image/jpeg")

    return app
