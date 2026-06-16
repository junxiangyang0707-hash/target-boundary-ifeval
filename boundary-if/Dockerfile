FROM python:3.11-slim

ENV PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]"

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

CMD ["bash"]
