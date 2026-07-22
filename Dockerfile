FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 1000 framefeed \
    && useradd --uid 1000 --gid framefeed --create-home framefeed

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir .

USER framefeed
ENTRYPOINT ["framefeed"]
