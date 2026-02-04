FROM python:3.11-slim

WORKDIR /app

# Install TA-Lib C library (required for python TA-Lib wrapper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get purge -y --auto-remove build-essential wget \
    && rm -rf /var/lib/apt/lists/*

COPY ./app /app/app
COPY ./requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

# Bundle Swagger UI v5 assets locally (supports OpenAPI 3.1).
# This avoids relying on the browser being able to reach a CDN at runtime.
ARG SWAGGER_UI_DIST_VERSION=5
ENV SWAGGER_UI_DIR=/opt/swagger-ui
RUN python - <<'PY'
import os
import pathlib
import urllib.request

version = os.environ.get("SWAGGER_UI_DIST_VERSION", "5")
base = f"https://cdn.jsdelivr.net/npm/swagger-ui-dist@{version}/"

target = pathlib.Path(os.environ.get("SWAGGER_UI_DIR", "/opt/swagger-ui"))
target.mkdir(parents=True, exist_ok=True)

files = [
    "swagger-ui-bundle.js",
    "swagger-ui.css",
    "favicon-32x32.png",
]

for name in files:
    url = base + name
    dest = target / name
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded empty file: {dest} from {url}")

print(f"Swagger UI assets downloaded to {target}")
PY

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
