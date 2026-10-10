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

## The simulated world

`world/stream.py` generates the data. A seed fixes everything: which features
matter, the shape of the true decision boundary, and the rows drawn at each
step. There are six features, two of which carry no signal, and four classes,
one of them rare. The boundary has a small quadratic term, so a linear model
can get close to it but cannot fit it exactly.

Each world is then calibrated so that the best possible classifier reaches the
same accuracy and the classes keep the same frequencies, whatever the seed.
Two seeds give two differently shaped worlds that are equally hard.

The rows of a step depend only on the seed and the step number. Any step can be
regenerated on its own, and adding a new source of randomness later will not
change rows that were already drawn.

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs the Python version
pinned in `.python-version` and exactly the packages listed in
`requirements.txt`.

```bash
make env     # create the environment
make test    # run the tests
make help    # list the targets
```

The environment is created in `~/.venvs/driftgate`, outside the repository. On
WSL2 the repository can live on a Windows drive, but a virtualenv there is very
slow because it is made of thousands of small files. Set `VENV=...` to put it
somewhere else.

## License

MIT, see [LICENSE](LICENSE).
