# the environment only: interpreter and requirements, set up the way make env
# does it on any machine. the repository is mounted at /work when it runs, git
# history and all, so the code is whatever commit is checked out.
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587

COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git make \
    && rm -rf /var/lib/apt/lists/*

ENV VENV=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python
WORKDIR /env
COPY Makefile .python-version requirements.txt ./
RUN make env

# the mounted repository belongs to another user, git refuses to read it otherwise
RUN git config --system --add safe.directory /work
WORKDIR /work
ENTRYPOINT ["make"]
CMD ["test"]
