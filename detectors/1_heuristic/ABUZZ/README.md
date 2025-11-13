
# Mosquito Audio Preprocessing (ABUZZ)

This repository contains a small pipeline to preprocess **ABUZZ** recordings and extract **1-second mosquito-sound snippets**. It was designed for building datasets for downstream modeling (classification, detection, self-supervised learning, etc.).

The code is available both as a Jupyter notebook and a plain Python script:

* `detect_mosquito_sounds_ABUZZ_clean.ipynb`
* `detect_mosquito_sounds_ABUZZ_clean.py`

---

## What it does (high level)

1. **Walk input folders** (only first-level subfolders are treated as classes/buckets).
2. **Load audio** files of supported types: `.wav`, `.mp4`, `.m4a`, `.amr`.
3. **Resample to 8 kHz mono** and **slice into 1.0 s windows** with **0.5 s hop**.
4. **Speech filtering (Silero VAD):** chunks that contain speech are diverted to a separate folder.
5. **Mosquito heuristic detection:**

   * Convert chunk to float32 and apply a **100 Hz high-pass** (Butterworth).
   * Compute **STFT** (`n_fft=512`, `hop_length=50`) → **magnitude spectrogram in dB**.
   * Split the spectrogram into **10 equal time segments**; for each segment:

     * Aggregate spectrum across time (max per frequency bin).
     * Focus on the **300–1500 Hz band**.
     * Find a local minimum → a subsequent **peak** → next local minimum; measure **dB drop**.
     * Accept a segment if **peak < 1500 Hz** and **drop > 15 dB**.
   * A chunk is considered **mosquito-positive** if **≥ 3** segments pass this test.
6. **Export all chunks** to 16 kHz / mono / 16-bit WAV with a consistent naming scheme:

   * positives → `…/<class>/...`
   * speech → `…/<class>_speech/...`
   * negatives → `…/<class>_not_selected/...`

---

## Folder structure

Input (example):

```
INPUT_ROOT/
  A/
    rec1.wav
    rec2.m4a
  B/
    session1/
      x.wav
```

Output (example):

```
OUTPUT_ROOT/
  A/                       # mosquito-positive chunks
  A_speech/                # chunks with detected speech
  A_not_selected/          # negative chunks
  B/
  B_speech/
  B_not_selected/
```

Each saved chunk is named as:
`<original_stem>_chunk<index>.wav` (e.g., `rec1_chunk37.wav`)

---

## Requirements

* **Python 3.9+** recommended
* **ffmpeg** installed on your system (required by `pydub`)
* Python packages:

  * `pydub`
  * `numpy`
  * `librosa`
  * `scipy`
  * `torch`
  * `silero-vad` (package providing `load_silero_vad` and `get_speech_timestamps`)

Install example:

```bash
pip install numpy librosa scipy pydub torch silero-vad
# Ensure ffmpeg is installed via your OS package manager (apt, brew, choco, etc.)
```

---

## How to use

### 1) Notebook

Open `detect_mosquito_sounds_ABUZZ_clean.ipynb` and set the two parameters at the top:

```python
INPUT_ROOT = "./ABUZZ"                        # path to your input root
OUTPUT_ROOT = "./ABUZZ_preprocessed_database" # path for processed chunks
```

Then run all cells:

* Imports are right after the parameters.
* The last cell initializes Silero VAD and starts processing.

### 2) Script

Edit the top of `detect_mosquito_sounds_ABUZZ_clean.py` to set:

```python
INPUT_ROOT = "./ABUZZ"
OUTPUT_ROOT = "./ABUZZ_preprocessed_database"
```

Run:

```bash
python detect_mosquito_sounds_ABUZZ_clean.py
```

---

## Parameters & knobs

* **Chunking**

  * Window size: **1.0 s**
  * Hop size: **0.5 s**
* **Resampling**

  * Chunking at **8 kHz** mono for speed.
  * All exports enforced to **16 kHz / mono / s16** for consistency.
* **High-pass filter**: `cutoff=100 Hz`, `order=4`
* **STFT**: `n_fft=512`, `hop_length=50`
* **Heuristic thresholds**

  * Frequency band: **300–1500 Hz**
  * Post-peak drop: **> 15 dB**
  * Segment amplitude gate: **peak > 0.02** (skip very weak segments)
  * Chunk positive if **≥ 3** segments pass

> Tip: If you get too many false positives/negatives, adjust the **15 dB** drop, the **0.02** amplitude gate, or the **≥ 3** segment count. You may also tweak `hop_length` or the target band (e.g., 250–1400 Hz) depending on your microphones and species.

---

## Why this works (intuition)

Mosquito flight tones are relatively **tonal** and often sit in the **few-hundred to low-thousand Hz** range. In a short window, they produce **peaky spectral structure**. The heuristic checks for a **pronounced peak** within 300–1500 Hz and verifies that it **stands out** (a noticeable dB drop after the peak). Splitting the chunk into multiple time segments makes the detector more robust to **intermittent** buzzing and short artifacts.

Speech is filtered first with **Silero VAD** to avoid confusing voiced speech with tonal insect sounds and to keep speech separate for later inspection.

---

## Troubleshooting

* **No audio exported**
  Ensure `ffmpeg` is installed and the input formats are supported. Verify `INPUT_ROOT` points to the right folder and that subfolders contain audio.
* **Everything ends up in `_speech`**
  Check your environment noise or try lowering speech sensitivity (you can modify the VAD call if needed).
* **Performance issues**
  Increase `hop_length` (e.g., 64 or 80), reduce the number of segments (e.g., 8), or skip very quiet chunks earlier.

---

## Notes & limitations

* This is a **heuristic** prefilter — it won’t be perfect. It’s intended to create a reasonable starting dataset for later ML/DNN training.
* The chosen band and thresholds are **data-dependent**; calibrate for your microphones, environments, and species if necessary.
* If you plan to train a neural model afterward, keep the `_speech` and `_not_selected` sets — they’re useful for **hard negatives**.

---

## License & Attribution

* Uses **Silero VAD** via `silero-vad` for speech detection.
* Requires **ffmpeg** with `pydub` for file I/O and resampling.
