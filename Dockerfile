FROM mambaorg/micromamba

USER root
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /app/environment.yml
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes

COPY animaloc animaloc
COPY configs configs
COPY notebooks notebooks
COPY tools tools

EXPOSE 8888
CMD ["jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]