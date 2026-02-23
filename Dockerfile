FROM python:3.13-slim

ARG TORCH_CPU

RUN groupadd -g 1000 fb2voice && useradd -m -u 1000 -g 1000 fb2voice

USER fb2voice
WORKDIR /app
ENV PATH="/home/fb2voice/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ONLY_BINARY=:all:

COPY --chown=fb2voice:fb2voice pyproject.toml .
RUN test -n "${TORCH_CPU}" && index_url="--index-url https://download.pytorch.org/whl/cpu" || index_url="" \
    && python -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['torch']))" \
        | xargs -r pip install ${index_url} \
    && python -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" \
        | xargs -r pip install --no-binary=docopt \
    && { pip uninstall -y setuptools wheel build || true; }

COPY --chown=fb2voice:fb2voice src ./src

ENTRYPOINT ["python", "-m", "fb2voice_ru.main"]