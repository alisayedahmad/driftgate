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

`world/truth.py` judges a model the way the pipeline never can. It draws fresh
rows from the world as it stands at a given step, from random numbers the
stream never uses, and scores the model against the true P(y|x) rather than
against drawn labels, so the only noise left is in the inputs. Under label
noise the truth is the clean one, and every model judged at the same step is
scored on the same rows.

## Labels

`world/labels.py` decides, for every row, whether its label ever comes back
and when. By default three rows in ten get one, after a delay of about ten
steps at the median with a long tail, and never during the step the row was
served. The draw only depends on the seed and the step, so a scenario and its
twin get the same labels at the same steps.

## Looking at a scenario

```bash
make world SEED=7 KIND=noise
```

prints what the scenario changes and, before and after the change, the best
possible accuracy, the accuracy of the model that was best before the change
(its real accuracy, and the one measured on the observed labels of every row as
if they all came back at once), and the share of each label. It shows the
world, not what the pipeline can know: there, only some labels come back, and
late. The same numbers go per step to `results/world/` as a
CSV, next to a plot of them.

## Models

`retrain/train.py` fits a multinomial logistic regression, each row counting
for its weight, and saves it as bytes. The same data and config give the same
bytes, in any process and with any number of threads, so a model can be known
by the hash of those bytes.

## Traceability

`spine/store.py` keeps bytes under the sha256 of their content, in a single
SQLite file. Storing the same thing twice keeps one copy, and reading checks
the bytes against their hash, so a damaged artifact is refused rather than
loaded.

`spine/provenance.py` names everything else a model depends on. A dataset is
the sha256 of its rows plus the recipe that regenerates them: a scenario, a
range of steps, the labeling, and the step it was known at, so that only rows
whose label had come back by then are in it. When its rows don't all weigh the
same, the recipe says which steps weigh what. The rows never need to be kept
and can be checked again at any time. The code is the commit, whether the working tree had changes, and a
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

fits a model at an early step of a scenario and another one some time after
its change, registers both with the first as the second's parent, and prints
the lineage. A model only uses what was known at the step it's trained at: the
labeled rows of the steps before it, the most recent of them held out to test
it on. Then it rebuilds each model from its record alone: the datasets are
regenerated from their recipes and checked against their hashes, the model is
refit with the stored config and has to come out with the same bytes, and the
metrics have to come out the same. Each run starts a fresh registry in
`~/.local/share/driftgate/`.

## Serving

`serving/log.py` keeps every prediction that was served: the step, the inputs,
the model that answered and its probabilities. A label only enters the log at
the step it comes back, and asking what was known at an earlier step hides the
labels that came later, so nothing can read a label before it existed.

`serving/app.py` answers with one model from the registry and refuses any
model that isn't registered. Replaying a scenario through it serves each step
in turn while the labels travel back as `world/labels.py` decided, entering
the log only when they arrive.

```bash
make serve SEED=7 KIND=concept
```

trains a model on what was known at an early step of the scenario, registers
it, and serves it at `http://127.0.0.1:8000`. `POST /predict` with
`{"step": 0, "rows": [[six numbers], ...]}` answers the probabilities over the
five classes, and every request lands in the log, so steps have to go forward.
HTTP is only a thin layer around the service; the rest of the pipeline uses the
service directly.

```bash
make replay SEED=7 KIND=concept
```

trains a model at an early step and replays the rest of the scenario through
the service. For each step it puts the true accuracy of what was served next
to the accuracy the pipeline could know at that step, from the labels that had
come back by then, and says how long after the change the known accuracy got
halfway down. It writes a CSV and a plot to `results/replay/`, and keeps the
log next to the run's registry.

## Monitoring

`monitor/cusum.py` is the detector every signal will use: a two-sided CUSUM
over standardized values, one stream per feature or class, updated once a
step. Its threshold trades how soon a real shift is caught against how often
it fires for nothing, and the tests check both against Siegmund's
approximation.

Monitors follow the log as a scenario is replayed: after each step they read
the rows just served and the labels that just came back, nothing else.

`monitor/input_drift.py` watches the mean of each feature, step by step,
against every row of the window the model was trained on, labeled or not,
regenerated from its record. A concept break or label noise leaves the inputs
exactly as they were, so this monitor can't see either of them: its path is
bit for bit the one of the scenario's twin.

`monitor/prediction_drift.py` watches the share of each predicted class, step
by step, against what the model predicted on that same window. It sees a
covariate shift or a new class within a few steps, and it is just as blind as
the input monitor to a concept break or to label noise: the inputs are the
same and so is the model, so the predictions are too.

`monitor/performance.py` reads the labels as they land. One monitor watches
their error rate against the accuracy the model was tested at; it sees a
concept break, late, and it also fires on label noise, where the model hasn't
got any worse. The other watches the share of each class among them against
the labels known at training time, and the first label of a class never seen
before is an alarm on its own. A concept break leaves those shares where they
were and label noise moves them, which is how the two can be told apart.

Every monitor also says where the change behind an alarm started: the step its
largest CUSUM sum last rose from zero, which is Page's estimate. If that sum
was already off zero when the change came, the estimate lands before it. The
two label monitors count in the steps labels come back at, so they place the
start a few steps after the rows the change hit.

```bash
make monitor
```

runs every kind of scenario on seeds set aside for tuning, with all four
monitors watching, and for each monitor and threshold measures how often it
fires on runs where nothing changed against how many changes of each kind it
catches, how late, and how far from the real change it puts the start. It
prints the thresholds that keep false alarms under a budget and writes the
whole trade-off to `results/monitor/`. No threshold is good at both, which is
the point.

## Retraining

`retrain/trigger.py` turns the alarms into one decision about the serving
model. The naive policy retrains as soon as the errors go up. The informed one
looks at what else has moved by then: if the inputs or the predictions did, the
world changed and it retrains; if only the shares of the labels did, the labels
changed and not the model, so it holds; if nothing else did, the concept
changed and it retrains. Under label noise the errors sometimes go off before
the label shares do, and then it retrains anyway. Inputs that move while the
errors stay put call for nothing, and a label of a class the model never saw
means retraining once enough of them are back to learn it from.

`retrain/window.py` chooses what the new model learns from. It waits a few
steps after the decision for more labels to come back, holds out the most
recent steps to test the new model on, and cuts the rest a little before where
the change was estimated to start. What came before the cut can be dropped,
kept at a lower weight, or kept in part, the steps nearest the cut, in case the
alarm was wrong. Each step since the change goes either to training or to the
test set, and with labels coming back late neither gets many.

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
