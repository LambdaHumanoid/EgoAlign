# Object Relocation sample data

`data/basket/` contains one Object Relocation episode and its two camera videos:
compact episode zero, numerical body/hand observations, minimal LeRobot metadata,
existing camera-alignment results and a checksummed `sample_manifest.json`.
The recorded instruction is `Move the basket to the table`.

## Content and integrity

The sample has 1009 raw frames; the common camera interval selects 1006 frames.
The MP4 files contain video-only streams. Camera offsets and durations are
provided in `sync/`; preserve them when copying the dataset.

Run `python tools/audit_source.py` to verify the files against
`sample_manifest.json`. Keep the manifest with the sample.

These are raw reproduction inputs. The [recipe](README.md) reconstructs causal
state and motion-token supervision through actual controller replay.

