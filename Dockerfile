FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV OCR_ENABLED=true
ENV OCR_LANGUAGE=spa

# Install local OCR and PDF processing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-spa \
    ocrmypdf \
    ghostscript \
    qpdf \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY normative_repository ./normative_repository

# Create data directories
RUN mkdir -p \
    /app/data/incoming_originals \
    /app/data/reports \
    /app/data/extracted \
    /app/data/markdown \
    /app/data/structured \
    /app/data/index

ENV DATA_DIR=/app/data
ENV SOURCE_DIR=/app/data/incoming_originals

ENTRYPOINT ["python", "-m", "normative_repository"]
CMD ["--help"]