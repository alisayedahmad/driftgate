# driftgate

A model in production slowly goes out of date as the data changes. The usual
answer is a pipeline that watches for drift, retrains, and swaps in the new
model when it looks better. This project is about checking whether that
pipeline makes the right calls: does it raise the alarm when it should, does it
retrain for good reasons, does it ever promote a model that is actually worse?

To answer that, the data is simulated. A scenario changes the data at a known
step and in a known way, so every decision the pipeline takes can be compared
with what really happened.

Early stage: the scenario generator comes first.

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs the Python version
pinned in `.python-version` and exactly the packages listed in
`requirements.txt`.

```bash
make env     # create the environment
make help    # list the targets
```

The environment is created in `~/.venvs/driftgate`, outside the repository. On
WSL2 the repository can live on a Windows drive, but a virtualenv there is very
slow because it is made of thousands of small files. Set `VENV=...` to put it
somewhere else.

## License

MIT, see [LICENSE](LICENSE).
