# driftgate

A model in production slowly goes out of date as the data changes. The usual
answer is a pipeline that watches for drift, retrains, and swaps in the new
model when it looks better. This project checks the calls such a pipeline
makes: when it raises an alarm, when it retrains, and whether the model it
promotes is really better.

To answer that, the data is simulated. A scenario changes the data at a known
step and in a known way, so every decision the pipeline takes can be compared
with what really happened.

Early stage: the scenario generator comes first.

## The simulated world

`world/stream.py` generates the data. A seed fixes how much each feature
matters, the shape of the true decision boundary, and the rows drawn at each
step. There are six features, the last two carry no signal, and four classes,
one of them rare. The boundary has a quadratic term, so a linear model can't
fit it exactly.

Each world is then calibrated so that, whatever the seed, the best possible
classifier has the same accuracy and the classes have the same frequencies.
The shape still varies from seed to seed, and so does how well a given model
does on it.

The rows of a step depend only on the seed and the step number. Any step can be
regenerated on its own, and adding a new source of randomness later will not
change rows that were already drawn.

## Scenarios

`world/scenarios.py` adds one change to a world: a seed, a kind of change, the
step where it happens and how strong it is. Before that step the rows are
exactly those of the stationary world.

Every scenario has a twin with the same seed and no change. Both use the same
random numbers, so their rows can be compared one by one and whatever differs
comes from the change.

Kinds so far:

- `none` is the control, nothing happens.
- `covariate` moves the mean of one feature. How labels depend on the inputs
  stays the same. If the feature is one of the two without signal, the inputs
  drift and every label stays what it would have been.
- `concept` changes how labels depend on the inputs: the informative features
  are rotated before the labels are drawn. The inputs don't move, and since
  they are spread evenly in every direction, the class frequencies and the best
  possible accuracy don't either. The size of the change is the accuracy the
  old best classifier loses.
- `noise` relabels part of one class as another class from the change on. The
  inputs and the true link between inputs and labels don't move, only the
  labels the pipeline gets to see. Measured against them, a model that is
  still right usually looks worse. Retraining on them makes a model worse on
  the true labels, a little overall and mostly on the relabeled class.
- `new` brings in a fifth class from the change on. Its rows sit in a part of
  the informative features that was almost empty before, and the other rows
  don't change. A model trained before can't predict that class, so it gets
  all of those rows wrong, usually by picking one of the classes it knows with
  high confidence.

## Setup

You need `make` and [uv](https://docs.astral.sh/uv/). uv installs the Python
version pinned in `.python-version` and exactly the packages listed in
`requirements.txt`.

```bash
make env     # create the environment
make test    # run the tests
make help    # list the targets
```

The environment is created in `~/.venvs/driftgate`, outside the repository. On
WSL2 the repository can live on a Windows drive, but a virtualenv there is
slow, it's lots of small files. Set `VENV=...` to put it somewhere else.

## License

MIT, see [LICENSE](LICENSE).
