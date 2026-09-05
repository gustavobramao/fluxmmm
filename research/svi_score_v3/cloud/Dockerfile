FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY research/svi_score_v3/cloud/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.lock

COPY scripts/mcmc_server.py scripts/svi_batch.py scripts/cloud_svi_task.py /app/scripts/

ENTRYPOINT ["python", "/app/scripts/cloud_svi_task.py"]
