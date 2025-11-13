
# Mosquito Sound Detection for Setup-1

The main detection script for this project is `detect_mosquito_sounds_to_csv-setup_channel_2_en.ipynb`. It provides a Python script (runnable in a Jupyter Notebook) to detect mosquito sound events within audio recordings made by setup-1 (discussed in our paper). It processes `.wav` files, identifies 1-second segments containing potential mosquito wing-beat sounds, and outputs both a CSV file with timestamps and the corresponding audio segments.

This script is designed to be flexible:

  * If a file's channel count doesn't match the `channel_num` parameter, the script will print a warning and process all available channels.
  * Providing a noise profile via `noise_profile_fn` is **optional but highly recommended** for accuracy. If a profile for a specific file/channel is not found, the script prints a warning and proceeds without noise reduction.

## Dependencies

Before running, ensure you have Python 3 and the following libraries installed:

  * **`ffmpeg` (system-level):** This is not a Python library, but a command-line tool.
      * **Why is this needed?** The `pydub` library (which is used for loading, splitting, and resampling audio) requires `ffmpeg` to be installed on your system as its backend. It handles the actual audio file decoding and encoding.
  * **Python Packages:**
      * `pydub`
      * `numpy`
      * `librosa`
      * `scipy`
      * `pandas`
      * `torch`
      * `silero-vad`

You can install the required Python packages using pip:

```bash
pip install pydub numpy librosa scipy pandas torch silero-vad
```

-----

## Configuration

All main parameters are set in the **second code cell** of the `detect_mosquito_sounds_to_csv-setup_channel_2_en.ipynb` notebook:

  * `input_root = "./input_recordings"`: The folder path containing your input `.wav` recordings.
  * `output_root = "./processed_segments_csv"`: The directory where all outputs (CSV files and detected `.wav` segments) will be saved.
  * `channel_num = 2`: The *expected* number of channels. The script will print a warning if a file's channel count doesn't match this, but it will still process all channels found.
  * `noise_profile_fn = "noise_profiles_25_01_14.zip"`: **(Recommended)** The path to a pre-computed noise profile, (e.g., a `.zip` or `.csv` file readable by `pandas.read_csv`). This file is **optional**.
  * `b_draw = False`: A boolean flag. If set to `True`, the script will generate and display plots for debugging the spectrogram analysis.

**Note on VAD:** The script also imports `silero-vad` for voice activity detection. However, the function call `filter_speech` is currently **commented out** in the main processing loop (`anal_chunks`) and is therefore inactive.

-----

## How to Run

1.  **(Recommended) Prepare Noise Profile:** Run the `compute_noise_profiles_en.ipynb` script (see description below) on your dataset to generate the `noise_profile_fn` file.
2.  **Place Recordings:** Put your `.wav` files (any channel count) into the `input_root` folder (e.g., `./input_recordings`).
3.  **Configure Paths:** Open the `detect_...ipynb` notebook and verify that `input_root`, `output_root`, and `noise_profile_fn` point to the correct locations.
4.  **Run Notebook:** Execute all cells in the Jupyter Notebook.

-----

## Core Logic: How Mosquito Segments are Detected

1.  **Audio Splitting:** The script loads an audio file. It prints a warning if the file's channel count doesn't match `channel_num` but proceeds to split the audio into its separate mono channels regardless.
2.  **Chunking:** Each mono channel is resampled to **16,000 Hz** and sliced into **1-second (1000ms) chunks**, with a **0.5-second (500ms) overlap**.
3.  **Preprocessing:** Each 1-second chunk undergoes:
      * A **250 Hz high-pass filter** to remove low-frequency rumble.
      * Spectrogram calculation (STFT).
      * **Noise Reduction (Conditional):** The script attempts to find a matching noise profile for the file and channel from the `noise_profile_fn` file.
          * If found, it's used to denoise the spectrogram.
          * If not found, a warning is printed, and the analysis continues on the non-denoised (raw) spectrogram.
      * **Detrending:** A secondary filter is applied to remove the general "shape" or background hum from the spectrogram, making transient peaks more prominent.
4.  **Sub-Segment Analysis:** The 1-second spectrogram is divided into **10 smaller sub-segments**.
5.  **Detection:** The script analyzes each sub-segment for a specific acoustic signature in the **300-1500 Hz** range:
      * It first checks if the segment's raw amplitude is above a minimum threshold (`y_segment_max > 0.02`) to avoid analyzing pure silence.
      * It then searches for a prominent frequency peak followed by a significant dip (a `db_difference` greater than the `threshold` of 17). This peak-dip signature is characteristic of a harmonic sound (like a wing beat) standing out from broadband noise.
6.  **Thresholding:** A 1-second chunk is classified as a "mosquito sound" **if at least 3 of its 10 sub-segments** trigger the detection logic (`mosquito_count >= 3`).

-----

## Inputs and Outputs

### Inputs

  * **Audio Files (`input_root`):** A folder (e.g., `./input_recordings`) containing `.wav` files. The script will process all channels in each file.
  * **Noise Profile (`noise_profile_fn`):** **(Optional)** A single `.zip` or `.csv` file (e.g., `noise_profiles.zip`) that- contains the average noise spectrum for *each channel* of *each file*. If a file or channel is missing from this profile, it will be processed without noise reduction.

### Outputs

All outputs are saved to the `output_root` directory (e.g., `./processed_segments_csv`).

1.  **Detection CSVs:** For *each* input file processed, a corresponding CSV is created (e.g., `Test0001.csv`).
      * **If detections are found:** The CSV will list the metadata (filename, channel, start\_ms, end\_ms) for *only* the 1-second chunks that were positively identified.
      * **If no detections are found:** An *empty* CSV with only a header row is created.
2.  **Audio Segments (`output_root`):** The script also saves the actual 1-second `.wav` audio chunks that passed the detection.
      * *Example:* If `Test0001.csv` lists a detection for channel 1 at 500ms, the file `./processed_segments_csv/Test0001_1_500.wav` will be created.
3.  **Other Segments:** Two other folders are created:
      * `..._not_selected`: Can be used to save segments that *failed* the check.
      * `..._speech`: Can be used to save segments identified as speech by VAD (currently inactive).
4.  **(Optional) Saving Negative Samples:** The script can also save the 1-second chunks that *did not* pass the mosquito detection (i.e., negative samples). To enable this, you must uncomment the following line in the `anal_chunks` function:
    ```python
    #chunk.export(not_selected_file, format="wav", parameters=["-ar", "16000", "-ac", "1", "-sample_fmt", "s16"])
    ```
    These segments will be saved in the `..._not_selected` folder.

-----

## Noise Profile Generation (compute\_noise\_profiles\_en.ipynb)

This companion notebook is used to generate the `noise_profile_fn` file (e.g., `noise_profiles_.csv`) required by the main detection script for effective noise reduction.

### Purpose

The script iterates through all `.wav` files in its `input_folder`, splits them into individual channels, and computes a characteristic noise profile for each channel.

### How it Works

The noise profile is calculated for each channel by:

1.  Applying a **250 Hz high-pass filter** to the audio.
2.  Calculating the STFT (spectrogram) of the *entire* channel's audio.
3.  Taking the **median** value of each frequency bin across the entire file. This median vector represents the stable background noise signature of the recording setup for that specific channel.

The final output is a single CSV file (`output_csv`) that lists the noise profile values (median amplitude) for each frequency bin, per channel, per file.

### Configuration

Key parameters in this notebook:

  * `input_folder = "./input_recordings/"`: The path to the recordings (should be the same as `input_root` in the main script).
  * `output_csv = "noise_profiles_.csv"`: The name of the resulting CSV file.

### Dependencies

This script requires `pydub`, `numpy`, `pandas`, and `librosa`.
