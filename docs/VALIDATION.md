# Validation

Check both the source package and the replay outputs. Static tests do not
replace a controller run in your configured environment.

## Check the source

From the EgoAlign checkout, using the data environment:

```bash
python -m pytest -q
python tools/check_cli.py
python tools/audit_source.py
```

Tests cover path safety, camera correspondence, input shapes, finite values,
capture timing, subprocess failure and artifact identity. The source audit
verifies sample/media hashes and rejects symlinks and unapproved files. It is
not a complete security audit.

## Validate a processed episode

Run the [sample recipe](../examples/README.md) with the
[configured runtime](public_runtime.md), then validate the final dataset against
the accepted round's control capture:

```bash
"$EGOALIGN_DATA_PYTHON" -m egoalign.verify_sample \
  --stage-a /path/to/result/stage_a --final /path/to/result/final \
  --control /path/to/selected-replay/telemetry/episode_000000.control.npz \
  --output /path/to/result/payload_verification.json
```

The validator checks finite 43D states, exact same-tick state/token payloads,
unchanged hand labels, timestamps and camera hashes. The processing pipeline
also checks rollout identity, timing and acceptance. Keep its command logs,
configuration, source/asset hashes and telemetry with the output. Never replace
missing captures with offline staging tokens.

## Interpret the outputs

- Select the replay recorded in `final_decision.json` (`accepted_round`);
  early stopping means it is not always Round 2. Replay-only uses `baseline_replay`.
- Read measured endpoint errors from replay evaluation, not candidate FK error.
  Record the active-frame selection, temporal compensation and units when
  comparing runs. README metrics refer to their accompanying demonstration videos.
- Inspect `render_report.json` after [video export](../examples/README.md#export-the-final-mujoco-video)
  for frame counts and pose-fidelity checks.
- Endpoint acceptance and payload validation do not certify physical task success.
  Calibrate production quality profiles for your data and check their acceptance
  separately; `production_accepted=false` must not be treated as production approval.

Run the sample in each new dependency environment. Source checks and recorded
dependency versions alone do not establish working controller dynamics or a
clean-machine installation. See [dependencies](DEPENDENCIES.md) and
[external assets](EXTERNAL_ASSETS.md).

