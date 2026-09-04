FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_CACHE=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra app --no-editable \
    && useradd --uid 10001 --create-home eovr
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
ENTRYPOINT ["/app/.venv/bin/eovr", "serve", "--host", "0.0.0.0"]
