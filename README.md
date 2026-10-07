# CA4 convolution with explicit im2col

Convolution is implemented with explicit image-to-column windows and matrix multiplication. Column-to-image accumulates overlapping contributions. Pooling uses explicit windows. Reference tests check values and all input/weight/bias gradients. The 29 ordered tabular features are standardized and zero-padded into a 7×7 grid; this is an artificial feature arrangement, not a natural-image benchmark.

Completed academic assignment with a portable experiment, local checkpoint prediction, independent algorithm tests and fresh held-out evaluation. See `ATTRIBUTION.md` for supplied/assisted source boundaries. This is not original research or a production classifier.

## Run

Use Python 3.12, install `requirements.txt`, then run:

```text
python -m pytest -q
python experiment.py train --train-csv /path/to/train.csv --output-dir /path/to/local-results
python experiment.py predict --checkpoint-dir /path/to/local-results --predict-csv /path/to/unlabelled.csv --output-csv /path/to/predictions.csv
```

Training requires the assignment's locally retained CSV, ordered numeric columns `f0` through `f28`, followed by `label`. Prediction uses the same ordered features without a label column. Non-finite/missing features, insufficient classes and conflicting duplicate vectors are rejected. Exact duplicate feature vectors are removed before splitting. The dataset fingerprint is recorded in `metrics.json`. Data, predictions, model binaries and media stay local; no dataset or weights are embedded here. Only load your own trusted local checkpoints.

## Evaluation

All experiments use a fixed seed of 42 and a stratified 60% training / 20% validation / 20% held-out test split. Scaling is fitted on training features only. Validation cross-entropy selects the checkpoint; the test labels never select an epoch or model. Epoch caps and architecture settings were fixed before these runs. Torch models stop after 12 unimproved validation epochs. The NumPy model runs its fixed cap of 200 epochs. `--epochs` can explicitly change the cap for your own run. Comparison variants use the same initialization seed and sample order, though their training dynamics differ.

Fresh results on the locally retained coursework training dataset:

| Model | Selected epoch | Held-out accuracy | Macro F1 |
|---|---:|---:|---:|
| MyCNN | 3 | 37.08% | 0.371 |

These are one-split academic results, not leaderboard scores, multi-seed confidence estimates or evidence of broad deployment performance. The official unlabelled test/private files have no available ground truth, so no accuracy is claimed for them. Class balance, split counts, confusion matrices, validation histories and source SHA-256 are in `metrics.json`. Saved checkpoints were reloaded and produced identical held-out probabilities.

## Notebook and verification

`assignment.ipynb` follows the setup → implementation check → experiment → results pattern. It runs a mathematical smoke check and reads the included real metrics; full retraining uses the command above with local data. Published notebook outputs are cleared. Tests check the underlying mathematics against independent references and validate split separation, train-only scaling, rejected data, checkpoint reload and the full prediction command. `VERIFICATION.json` records the executed checks.
