FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY environment.yml /tmp/environment.yml

RUN conda env update -n base --quiet --file /tmp/environment.yml \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"
