# driftgate

A model in production slowly goes out of date as the data changes. The usual
answer is a pipeline that watches for drift, retrains, and swaps in the new
model when it looks better. This project checks the calls such a pipeline
makes: when it raises an alarm, when it retrains, and whether the model it
promotes is really better.

To answer that, the data is simulated. A scenario changes the data at a known
step and in a known way, so every decision the pipeline takes can be compared
with what really happened.

Early stage: the scenario generator is there, the pipeline it's meant to test
comes next.

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

The tests keep a fingerprint of what a few seeds draw for every kind, in
`world/fingerprints.txt`. A change to the code that moves the data fails them
even when every other test passes. `make fingerprints` rewrites the file when
that change is meant.

## Looking at a scenario

```bash
make world SEED=7 KIND=noise
```

prints what the scenario changes and, before and after the change, the best
possible accuracy, the accuracy of the model that was best before the change
(its real accuracy and the one measured on the labels the pipeline sees), and
the share of each label. The same numbers go per step to `results/world/` as a
CSV, next to a plot of them.

## Models

`retrain/train.py` fits a multinomial logistic regression and saves it as
bytes. The same data and config give the same bytes, in any process and with
any number of threads, so a model can be known by the hash of those bytes.

## Traceability

`spine/store.py` keeps bytes under the sha256 of their content, in a single
SQLite file. Storing the same thing twice keeps one copy, and reading checks
the bytes against their hash, so a damaged artifact is refused rather than
loaded.

`spine/provenance.py` names everything else a model depends on. A dataset is
the sha256 of its rows plus the recipe that regenerates them, a scenario and a
range of steps, so the rows never need to be kept and can be checked again at
any time. The code is the commit, whether the working tree had changes, and a
hash of those changes. Configs are hashed as canonical JSON.

`spine/registry.py` keeps one record per model: the datasets it was trained
and evaluated on, its config, the code and library versions, its metrics, the
model that was serving when it was trained, and the simulation step. A record
can only point at things the store holds, and following the serving models
back gives a model's lineage. There is no clock time in a record, so two
identical runs give identical registries.

```bash
make train SEED=7 KIND=concept
```

fits a model on the first steps of a scenario and another one after its
change, registers both with the first as the second's parent, and prints the
lineage. Then it rebuilds each model from its record alone: the datasets are
regenerated from their recipes and checked against their hashes, the model is
refit with the stored config and has to come out with the same bytes, and the
metrics have to come out the same. Each run starts a fresh registry in
`~/.local/share/driftgate/`.

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
