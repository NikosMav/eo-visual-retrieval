"""HTTP routing for the comparison surface.

This is the only module here that knows about HTTP. All ranking decisions live
in catalog.py so they can be tested without a server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from eo_visual_retrieval.app.catalog import Catalog, ModelRanking
from eo_visual_retrieval.app.thumbnails import thumbnail_jpeg
from eo_visual_retrieval.app.uploads import MAX_UPLOAD_BYTES, decode_upload

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
THUMBNAIL_PIXELS = 192


def _headline(findings: dict[str, Any]) -> dict[str, Any]:
    """Derive the few figures the page states in prose, from the same payload.

    Writing these into the template by hand is exactly how a page drifts from
    the evidence it claims to summarize.
    """

    days = [place["days"] for place in findings["places"]]
    queries = findings["queries"] or 1
    return {
        "best_overlap": f"{findings['overlap_max'] * findings['k']:.0f} of {findings['k']}",
        "all_correct_pct": round(
            findings["top_one"]["all_stores_correct"] / queries * 100, 1
        ),
        "days_max": max(days) if days else 0,
        "days_min": min(days) if days else 0,
    }


class ContentLengthLimitMiddleware:
    """Bound actual request bytes before multipart parsing, including chunked bodies.

    Buffer at most the cap in memory, then replay the body to the parser. An
    oversized body never reaches the parser or its temporary-file spool.
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
                    length = -1
                if length < 0:
                    await PlainTextResponse("invalid content length", status_code=400)(
                        scope, receive, send
                    )
                    return
                if length > self._max_bytes:
                    response = PlainTextResponse("upload exceeds size limit", status_code=413)
                    await response(scope, receive, send)
                    return
            buffer = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(buffer) + len(chunk) > self._max_bytes:
                    await PlainTextResponse("upload exceeds size limit", status_code=413)(
                        scope, receive, send
                    )
                    return
                buffer.extend(chunk)
                if not message.get("more_body", False):
                    break
            body = bytes(buffer)
            buffer.clear()
            delivered = False

            async def replay() -> Message:
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}

            await self._app(scope, replay, send)
            return
        await self._app(scope, receive, send)


def create_app(
    catalog: Catalog, *, k: int = 5, findings: dict[str, Any] | None = None,
    product_navigation: bool = False,
) -> FastAPI:
    """Build the application around one already-validated catalog."""

    if k < 1:
        raise ValueError("k must be positive")
    if k > catalog.index_size:
        raise ValueError(
            f"k={k} exceeds the index size {catalog.index_size}: lower --k or "
            "supply a larger corpus"
        )
    app = FastAPI(title="EO visual retrieval")
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    app.add_middleware(ContentLengthLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)
    query_groups: dict[str, list[str]] = {}
    for item_id in catalog.query_ids:
        query_groups.setdefault(catalog.label(item_id) or "Unlabelled", []).append(item_id)

    def render(
        request: Request,
        rankings: list[ModelRanking],
        *,
        query_id: str | None,
        is_upload: bool,
        error: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="compare.html",
            status_code=status_code,
            context={
                "rankings": rankings,
                "query_ids": catalog.query_ids,
                "query_id": query_id,
                "is_upload": is_upload,
                "upload_available": catalog.upload_available,
                "query_groups": dict(sorted(query_groups.items())),
                "query_label": catalog.label(query_id) if query_id else None,
                "index_size": catalog.index_size,
                "k": k,
                "error": error,
                "findings_available": findings is not None,
                "product_navigation": product_navigation,
                "active": "models",
            },
        )

    @app.exception_handler(HTTPException)
    async def page_error(request: Request, error: HTTPException) -> Response:
        if request.url.path == "/thumbnail":
            return PlainTextResponse("thumbnail unavailable", status_code=error.status_code)
        return render(request, [], query_id=None, is_upload=False,
                      error=str(error.detail), status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_form(request: Request, error: RequestValidationError) -> Response:
        return render(request, [], query_id=None, is_upload=False,
                      error="Choose a query or an image file, then try again.", status_code=422)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    if findings is not None:
        # Registered only when evidence was supplied. A deployment without the
        # reports serves no findings rather than an empty or invented page.
        headline = _headline(findings)

        @app.get("/findings", response_class=HTMLResponse)
        def findings_page(request: Request) -> HTMLResponse:
            return TEMPLATES.TemplateResponse(
                request=request,
                name="findings.html",
                context={
                    "findings": findings,
                    "headline": headline,
                    # Embedded in a script block, so the template marks it safe
                    # and the escaping happens here: only "<" can end the block
                    # early, and escaping it keeps the JSON valid.
                    "findings_json": json.dumps(findings).replace("<", "\\u003c"),
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
        try:
            if not catalog.upload_available:
                raise ValueError("Image upload is unavailable in this demo. Choose a corpus query.")
            data = await image.read(MAX_UPLOAD_BYTES + 1)
            # decode_upload does real image-library CPU work; running it inline in
            # an async handler would block the event loop and stall every other
            # request for the duration of one decode.
            pixels = await run_in_threadpool(decode_upload, data, image_size=catalog.image_size)
            ranking = await run_in_threadpool(catalog.rank_uploaded, pixels, k=k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            await image.close()
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
