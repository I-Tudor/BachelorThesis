# MAJOR TOM — Roman Numeral Chord Analyzer

MAJOR TOM is a deep-learning system that performs **Roman Numeral Analysis (RNA)
directly from audio**. Given a polyphonic recording, it predicts a beat-level
harmonic analysis — global key, root scale degree, tonicization, chord quality,
inversion and the composite Roman numeral — and presents the result in a desktop
application with a synced waveform, chord timeline and lead-sheet PDF export.

The model (an `RNATransformer`) is trained and evaluated on the **PARC
(Polyphonic Audio to Roman Corpus)** dataset and reaches 76.3% global-key
accuracy and 49.9% strict Roman Numeral Conversion accuracy on the artist-level
test split.

> **Platform:** this project was developed and tested **only on macOS** (Apple
> Silicon, using the MPS backend). All instructions below assume macOS; other
> operating systems are not supported.

This README covers the four things you most likely came here for:

1. [Downloading and preparing the dataset](#1-dataset-setup)
2. [Editing the training configuration](#2-configuration)
3. [Training a model](#3-training)
4. [Running the desktop app](#4-running-the-app)

---

## Repository layout

```
RomanNumeralApp/
├── app/                     Desktop application (PyQt6)
│   ├── main.py              App entry point
│   ├── inference.py         Audio -> features -> predictions pipeline
│   ├── player.py            Audio playback
│   ├── export_pdf.py        Lead-sheet PDF export
│   └── ui/                  All Qt widgets (waveform, timeline, etc.)
├── source/                  Model + training library
│   ├── models/              RNATransformer (mine) + PARC baselines (not mine)
│   ├── data.py              TheoryTabDataset (HDF5-backed)
│   ├── loss.py              Multi-task loss
│   ├── metrics.py           Multi-task metrics
│   ├── augmentations.py     Pitch-transpose augmentation
│   └── constants.py         Tasks, label maps, and all file paths
├── scripts/
│   ├── extract_vamp_features.py   Build features.h5 from audio
│   ├── create_label_segments.py   Build labels.h5 from annotations
│   ├── train.py                   Training loop
│   ├── evaluate.py                Test-set evaluation
│   ├── demo_ui.py                 Run the UI with fake data (no model)
│   └── test_vamp_pitch.py         Check your Vamp pitch offset
├── configs/                 Training configuration YAMLs
├── dataset/
│   └── metadata/            label_domains.json, label_sizes.json, … (shipped)
├── segments/                Built feature/label HDF5 files go here
├── experiments/             Training output (checkpoints, metrics.csv)
└── requirements.txt
```

---

## Installation

Use Python 3.10+ and a virtual environment.

**1. Clone the repository**

> **Skip this step if you received the project as an archive** — just extract it
> and `cd` into the extracted `RomanNumeralApp` folder instead.

```bash
git clone https://github.com/I-Tudor/BachelorThesis
cd RomanNumeralApp
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install the Python dependencies**

```bash
pip install -r requirements.txt
```

This installs everything needed to train, evaluate, and run the app (PyTorch,
librosa, h5py, music21, wandb, scikit-learn, PyQt6, reportlab, and the rest).
The only dependency not covered here is the Vamp / NNLS-Chroma plugin, set up
next.

### The Vamp / NNLS-Chroma plugin (important)

Feature extraction uses the **NNLS-Chroma** Vamp plugin, exactly as the PARC
pipeline does. The Python `vamp` binding is installed for you by
`requirements.txt`, but the native plugin itself is *not* installed by `pip` and
must be added separately:

- Download the NNLS-Chroma plugin from <https://www.isophonics.net/nnls-chroma>
- Place the plugin files (`nnls-chroma.dylib` + `.cat`/`.n3`) in your macOS
  Vamp plugin folder: `~/Library/Audio/Plug-Ins/Vamp`

If the plugin is unavailable, the **app** falls back to librosa's `chroma_cqt`,
but **feature extraction for training requires the real plugin**.

#### Verify the pitch offset

On some systems the plugin returns chroma rotated by 3 semitones. The codebase
corrects this with `VAMP_SEMITONE_ROLL = -3`. Confirm your setup matches:

```bash
PYTHONPATH=. python scripts/test_vamp_pitch.py
```

If it reports `Peak bin: 3 (D#)`, the `-3` roll is correct (the default). If it
reports `Peak bin: 0 (C)`, set the roll to `0` in `source/constants.py`
(`VAMP_SEMITONE_ROLL`) and `app/inference.py`.

> **Note on running commands:** the scripts import the `source`/`app` packages
> and read paths *relative to the project root*. Always run commands from the
> repository root, and prefix with `PYTHONPATH=.` (as shown throughout) so the
> packages resolve correctly.

---

## 1. Dataset setup

Training data comes from **PARC (Polyphonic Audio to Roman Corpus)** —
popular-music tracks with beat-level harmonic annotations sourced from
Hooktheory/TheoryTab. See Poppe, Lopes & Figueiredo, *The Polyphonic Audio to
Roman Corpus*, DLfM 2025.

### Download

The dataset source is the official PARC project:

- **PARC repository:** <https://github.com/uai-ufmg/parc>
- **Dataset & features (Google Drive):**
  <https://drive.google.com/drive/folders/1zjEo_mlVFfvb6ouQ67sxk3RG4faeJOFp?usp=sharing>
- **Paper (ACM DL):** <https://dl.acm.org/doi/full/10.1145/3748336.3748345>

From the Google Drive folder you need two files:

- `features.h5` — the pre-extracted, pre-windowed NNLS-Chroma / Semitone-Spectrum
  features
- `labels.h5` — the pre-encoded, pre-windowed labels

(The folder also contains `parc.json`, the raw per-track annotations. It is not
required to train, evaluate, or run the app, so you can ignore it.)

Everything else is already local in this repository: the split/pitch-class files
(`dataset/splits.json`, `dataset/pcsets.json`), the label metadata
(`dataset/metadata/*.json`), and a trained checkpoint (`experiments/my_run/best_model.ckpt`).

> **License:** the PARC dataset and trained models are distributed under
> **CC BY-NC-SA 3.0** (derived from Hooktheory user contributions — see
> <https://forum.hooktheory.com/tos>). The code is MIT-licensed; the data is
> non-commercial.

### Placing the files

This project reads its data paths from `source/constants.py`. The two files from
the Drive need to be put where those constants expect them (defaults shown), or
edit the constants to point at your copies. Everything else already lives in the
repo:

| File | Constant | Default path | Source |
|------|----------|--------------|--------|
| `features.h5` | `FEATURES_FILEPATH` | `segments/features.h5` | **Drive** — place inside `segments/` |
| `labels.h5` | `LABELS_FILEPATH` | `segments/labels.h5` | **Drive** — place inside `segments/` |
| `splits.json` | `SPLITS_FILEPATH` | `dataset/splits.json` | Local (already in repo) |
| `pcsets.json` | `PCSETS_FILEPATH` | `dataset/pcsets.json` | Local (already in repo; used only by `evaluate.py`) |
| label metadata | — | `dataset/metadata/*.json` | Local (already in repo) |
| trained checkpoint | — | `experiments/my_run/best_model.ckpt` | Local (already in repo, or produced by training) |

Place the downloaded `features.h5` and `labels.h5` inside `segments/` and you are
ready to [train](#3-training). These are exactly the files `source/constants.py`
points to (`FEATURES_FILEPATH`, `LABELS_FILEPATH`) and that the dataset loader
reads.

---

## 2. Configuration

Training is driven entirely by a YAML file in `configs/`. The shipped example is
`configs/transformer_artist_chroma.yaml`:

```yaml
model: RNATransformer            # RNATransformer | Frog | BiGRU | AudioAugmentedNet | NaiveBaseline
run_name: "major-tom-full"       # name shown in Weights & Biases
num_epochs: 100

# Components of the final MAJOR TOM model
augment: true                    # pitch-transposition augmentation (train split only)
label_smoothing: 0.1             # uniform label smoothing on the non-key tasks
use_tonal_key_loss: true         # tonal-distance (circle-of-fifths) key smoothing
tonal_key_alpha: 0.15            # strength of the tonal-distance key smoothing
warmup_epochs: 10                # linear warmup before cosine decay

# Evaluated but disabled — did not improve results (see note below)
use_equiv_loss: false            # equivalence-aware (enharmonic) RN loss
add_chord_change_head: false     # auxiliary chord-change prediction head

output_dir: experiments/transformer_artist_chroma_aug/   # checkpoints + metrics.csv

data:
  split_level: artist            # artist | song | theorytab (artist = no artist leakage)
  use_semitone_spectrum: false   # false -> 12+12 chroma; true -> 84-bin spectrum

dataloader:
  num_workers: 0
  batch_size: 256

optimizer:
  lr: 0.0005
  weight_decay: 0.01

model_kwargs:                    # passed straight to RNATransformer(...)
  dropout: 0.4
  tome_r: 0                      # ToMe token merging disabled (no improvement)
  d_model: 128
  num_layers: 4
  nhead: 4
  dim_feedforward: 512
```

### Active parameters (the final model)

These settings reproduce the full MAJOR TOM reported in the thesis.

- **`model`** — which architecture to train. `RNATransformer` is the full MAJOR
  TOM model (my own work). The other options — `Frog`, `BiGRU`,
  `AudioAugmentedNet`, and `NaiveBaseline` — are the **PARC baseline models, taken
  from the PARC repository (<https://github.com/uai-ufmg/parc>); they are not my
  own work** and are included only for comparison. `NaiveBaseline` requires no
  training and exits immediately.
- **`augment`** — pitch-transposition data augmentation. Transposes each training
  example through the keys so the model learns key-invariant intervallic structure
  instead of memorising the skewed key distribution of pop music. This is the
  single largest contributor to accuracy in the thesis, so keep it on.
- **`label_smoothing`** — standard uniform label smoothing on the non-key
  classification heads; a mild regulariser that softens the one-hot targets.
- **`use_tonal_key_loss` / `tonal_key_alpha`** — tonal-distance label smoothing for
  the global-key head. It spreads a little probability mass onto keys that are
  close on the circle of fifths, so residual key errors land on tonally adjacent
  keys. `tonal_key_alpha` controls how much mass is redistributed.
- **`warmup_epochs`** — length of the linear learning-rate warmup before the
  cosine-annealing decay takes over.
- **`output_dir`** — where `best_model.ckpt` and `metrics.csv` are written. Give
  every run its own directory so checkpoints aren't overwritten. The app later
  loads `best_model.ckpt` from here.
- **`data.split_level`** — `artist` is the strict PARC protocol (no artist
  appears in two splits). Use `song` or `theorytab` only for looser experiments.
- **`data.use_semitone_spectrum`** — `false` feeds the dual 12-bin chroma streams
  (`in_channels = (12, 12)`); `true` switches to the 84-bin spectrum
  (`in_channels = (84,)`). This must match how you run the app.
- **`model_kwargs`** — transformer capacity. Lower `d_model` / `num_layers` for a
  smaller, faster model; `nhead` must divide `d_model`.

### Disabled parameters (left out — no measured improvement)

The codebase also implements several extra training components. They were
evaluated during development but **did not improve results**, so they are **not**
part of the final model described in the thesis and are turned off by default.
You can re-enable any of them to experiment, but expect no gain.

- **`use_equiv_loss`** — an equivalence-aware (enharmonic) training loss for the
  Roman-numeral head: it accepts enharmonically equivalent spellings as soft
  targets instead of a single hard label. (Enharmonic equivalence is used in the
  thesis only as an *evaluation* metric, not as a training objective.) Disabled.
- **`add_chord_change_head`** — an auxiliary head that predicts whether a chord
  change occurs at each frame, added as an extra multi-task signal. Disabled.
- **`boundary_weight` / `boundary_margin`** — boundary-upweighted loss that
  multiplies the cross-entropy of frames near chord transitions (with `*_margin`
  extra frames on each side), to push the model to commit at boundaries. Off by
  default (`boundary_weight: 1.0`); add the keys to enable.
- **`consistency_weight` / `consistency_tasks`** — within-segment consistency
  loss that penalises prediction variance between adjacent same-chord frames to
  reduce within-chord glitches. Off by default (`consistency_weight: 0.0`).
- **`model_kwargs.tome_r`** — ToMe (token merging) ratio; merges `r` transformer
  tokens per layer to cut compute. Kept at `0` (disabled).

To make a new experiment, copy the YAML, edit the fields, and point `--yaml` at
your copy:

```bash
cp configs/transformer_artist_chroma.yaml configs/my_experiment.yaml
```

---

## 3. Training

Training uses **Weights & Biases** for logging. Either log in once
(`wandb login`) or disable it for a local run (`export WANDB_MODE=offline`).

Run from the project root:

```bash
PYTHONPATH=. python scripts/train.py --yaml configs/transformer_artist_chroma.yaml
```

What happens:

- The device is auto-selected: Apple Silicon **MPS** if available, otherwise CPU.
- `TheoryTabDataset` loads the train and valid splits from
  `segments/features.h5` + `segments/labels.h5`.
- Each epoch trains, validates, steps the warmup->cosine LR schedule, logs to
  W&B, and appends a row to `<output_dir>/metrics.csv`.
- Whenever validation **`rn_conv_acc`** improves, a checkpoint is saved to
  `<output_dir>/best_model.ckpt`.

A run is reproducible (fixed seed 42). Reproducing the full model takes 100
epochs; reduce `num_epochs` for a quick smoke test.

### Evaluate a trained model

```bash
PYTHONPATH=. python scripts/evaluate.py --yaml configs/transformer_artist_chroma.yaml
```

This reports the strict and equivalence-aware metrics on the test split.
(`evaluate.py` additionally needs `dataset/pcsets.json`.)

---

## 4. Running the app

The desktop app loads a trained checkpoint and analyzes any audio file you open.

```bash
PYTHONPATH=. python app/main.py \
    --checkpoint experiments/my_run/best_model.ckpt \
    --label-domains dataset/metadata/label_domains.json \
    --label-sizes   dataset/metadata/label_sizes.json
```

Useful flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--checkpoint` | auto-detected from `experiments/` | Trained `.ckpt` to load |
| `--label-domains` / `--label-sizes` | from `dataset/metadata/` | Label vocabulary |
| `--device` | auto (mps -> cpu) | Force `cpu` or `mps` |
| `--nhead` | `4` | Attention heads — **must match the trained model** |
| `--bpm` | none | Override beat tracking with a fixed BPM |

Once the window opens, open an audio file from the UI; analysis runs on a
background thread and the chord timeline, waveform and detail panel populate when
it completes. Playback follows the timeline, and **Export Lead Sheet PDF**
(requires `reportlab`) writes a printable lead sheet.

### Demo / UI-only modes (no trained model needed)

```bash
# Random-prediction "dummy" model — just omit --checkpoint
PYTHONPATH=. python app/main.py

# Fully synthetic timeline + waveform, no model and no audio
PYTHONPATH=. python scripts/demo_ui.py
```

Both are handy for working on the interface before a checkpoint exists.

---

## Quick reference

```bash
# 0. Setup
pip install -r requirements.txt                         # then install the NNLS-Chroma plugin (see README)
PYTHONPATH=. python scripts/test_vamp_pitch.py          # verify pitch offset

# 1. Get the data: download features.h5 + labels.h5 from the Drive and
#    place them in segments/

# 2./3. Train (edit configs/*.yaml first)
PYTHONPATH=. python scripts/train.py --yaml configs/transformer_artist_chroma.yaml

# 4. Run the app
PYTHONPATH=. python app/main.py \
    --checkpoint experiments/my_run/best_model.ckpt \
    --label-domains dataset/metadata/label_domains.json \
    --label-sizes   dataset/metadata/label_sizes.json
```

---

## Troubleshooting

- **`ModuleNotFoundError: source` / `app`** — run from the repository root with
  `PYTHONPATH=.` prefixed.
- **`vamp` not found / wrong pitch** — install the NNLS-Chroma plugin into your
  Vamp path and run `scripts/test_vamp_pitch.py`; adjust `VAMP_SEMITONE_ROLL` if
  the peak bin isn't 3.
- **Predictions look transposed** — the Vamp pitch offset is misconfigured (see
  above).
- **App loads but predictions are garbage** — `--nhead` must match the trained
  model. The app only supports chroma-trained (12+12) checkpoints; if you trained
  with `use_semitone_spectrum: true`, that model cannot be loaded in the app.
- **W&B prompts on every run** — `export WANDB_MODE=offline` to train without an
  account.

