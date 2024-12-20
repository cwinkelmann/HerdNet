# Base image with CUDA 11.3, cuDNN 8, and Python 3.9
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Miniconda
RUN apt-get update && apt-get install -y \
    wget curl bzip2 git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -afy

# Add Conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment.yml into the container
COPY environment.yml /app/environment.yml

# Install Conda environment
RUN conda env create -f environment.yml

# Activate the environment for subsequent commands
SHELL ["conda", "run", "-n", "herdnet", "/bin/bash", "-c"]

# Install any additional dependencies if needed
RUN conda run -n herdnet pip install --no-cache-dir gdown==5.2.0

# Expose default Jupyter port
EXPOSE 8888

ENV JUPYTER_PASSWORD=123456
# Configure Jupyter dynamically
RUN mkdir -p /root/.jupyter && \
    python -c "from notebook.auth import passwd; \
               print(f'c.NotebookApp.password = \"{passwd(\"${JUPYTER_PASSWORD}\")}\"')" \
    > /root/.jupyter/jupyter_notebook_config.py

COPY data_FMO03_02_05 data/data_FMO03_02_05
COPY animaloc animaloc
COPY configs configs
COPY notebooks notebooks
COPY tools tools




# Add Conda and WORKDIR to PATH and PYTHONPATH
ENV PATH="/opt/conda/bin:$PATH"
# ENV PYTHONPATH="/app:$PYTHONPATH"

# Default command to start Jupyter Lab
CMD ["conda", "run", "-n", "herdnet", "jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]


# Now test with
# docker run --rm --gpus all herdnet-env nvidia-smi

# docker build -t herdnet-image . && docker run --gpus all -it --rm -p 8891:8888 --name cw_herdnet herdnet-image

# run the container
# docker run --gpus all -it -p 8888:8888 --name herdnet-env herdnet-cuda
