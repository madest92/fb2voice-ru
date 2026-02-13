FROM python:3.13-slim

ARG TORCH_CPU
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-suggests --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 fb2voice \
    && useradd -m -u 1000 -g 1000 fb2voice

USER fb2voice
ENV PATH="/home/fb2voice/.local/bin:$PATH"
WORKDIR /app

COPY --chown=fb2voice:fb2voice pyproject.toml .
RUN python -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" \
        | xargs pip install \
    && test -n "${TORCH_CPU}" && index_url="--index-url https://download.pytorch.org/whl/cpu" || index_url="" \
    && python -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['torch']))" \
        | xargs pip install ${index_url} \
    && mkdir -p /home/fb2voice/.cache/torch/hub/ \
    && python -c "import torch; torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v5_1_ru.pt', torch.hub.get_dir() + '/.silero_ru_v5_1.pt')"

COPY --chown=fb2voice:fb2voice src ./src

RUN pip install .

ENTRYPOINT ["fb2voice-ru"]
