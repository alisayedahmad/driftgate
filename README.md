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

## License

MIT, see [LICENSE](LICENSE).
