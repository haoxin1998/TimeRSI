FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.5.4 pandas==3.0.6 scipy==1.18.1 scikit-learn==1.9.1 lightgbm==4.7.0 datasets==5.1.0 fev==0.10.0 PyYAML==6.0.3 requests==2.34.2
# The CLI is supplied read-only at launch from the caller's licensed installation.
ENV PYTHONPATH=/opt/timersi/src PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
WORKDIR /workspace
