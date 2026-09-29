FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FLUX_OS_HOST=0.0.0.0 FLUX_OS_PORT=8000

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY flux_os ./flux_os
RUN pip install . && useradd --create-home --uid 10001 flux
USER flux

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"FLUX_OS_PORT\",\"8000\")}/health')"
ENTRYPOINT ["flux-os"]
CMD ["serve"]
