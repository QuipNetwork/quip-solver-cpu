# CI toolchain image for the Linux jobs: Rust 1.97.1 with clippy and rustfmt,
# the aarch64 cross target and linker, and a Python virtual environment with
# maturin, numpy and pytest for the binding in py/.
#
# The `build-toolchain-image` job in `.gitlab-ci.yml` builds and pushes this
# image. It runs automatically on a branch push that changes this file and is
# manual otherwise. The jobs pull the image by digest, so a rebuild takes
# effect only when a commit updates TOOLCHAIN_IMAGE.

# Nothing watches these pins for updates. Check the upstream releases on a
# rebuild: the rust image tags, and PyPI for the three Python packages.
FROM rust:1.97.1-trixie@sha256:b1b3c9c0d921d7fa0a6d1f9ec7e4eab87f8c8ec97644c3d791450f131dec813f

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN rustup component add clippy rustfmt \
    && rustup target add aarch64-unknown-linux-gnu

# python3-venv upgrades the image's python3.13 packages. That costs minutes in
# a job container, which is why it happens here, once.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc-aarch64-linux-gnu python3-venv \
    && rm -rf /var/lib/apt/lists/*

# `maturin develop` installs the binding into the active virtual environment,
# so the environment is on PATH and named in VIRTUAL_ENV.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir maturin==1.16.0 numpy==2.5.4 pytest==9.2.0

# Fail the image build here, not in a CI job.
RUN cargo --version && cargo clippy --version && cargo fmt --version \
    && rustup target list --installed | grep -qx aarch64-unknown-linux-gnu \
    && aarch64-linux-gnu-gcc --version | head -1 \
    && python --version && maturin --version && pytest --version \
    && python -c "import numpy; print('numpy', numpy.__version__)"
