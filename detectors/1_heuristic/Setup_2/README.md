
# Mosquito Sound Detection for Setup-2

The main detection script for this project is `detect_mosquito_sounds_to_csv-setup_channel_4_en.ipynb`. It provides a Python script (runnable in a Jupyter Notebook) to detect mosquito sound events within audio recordings made by setup-2 (4-channel recordings, discussed in our paper). It processes `.wav` files, identifies 1-second segments containing potential mosquito wing-beat sounds, and outputs both a CSV file with timestamps and the corresponding audio segments.

This script is specifically configured for **4-channel** audio files. Unlike other versions, this script **does not use pre-computed noise profiles**; it performs detection on the raw filtered spectrogram.

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

All main parameters are set in the **second code cell** of the `detect_mosquito_sounds_to_csv-setup_channel_4_en.ipynb` notebook:

  * `input_root = "./input_recordings"`: The folder path containing your input `.wav` recordings.
  * `output_root = "./preprocessed_database"`: The directory where all outputs (CSV files and detected `.wav` segments) will be saved.
  * `channel_num = 4`: The *expected* number of channels. The script will **skip** any file that does not have exactly 4 channels.

**Note on VAD:** The script actively uses `silero-vad` (via `filter_speech`) to detect and **skip** any chunks containing human speech. These chunks will not be processed for mosquito sounds.

-----

## How to Run

1.  **Place Recordings:** Put your **4-channel** `.wav` files into the `input_root` folder (e.g., `./input_recordings`).
2.  **Configure Paths:** Open the notebook and verify that `input_root` and `output_root` point to the correct locations.
3.  **Run Notebook:** Execute all cells in the Jupyter Notebook.

-----

## Core Logic: How Mosquito Segments are Detected

1.  **Audio Splitting:** The script loads an audio file. It checks if the file has exactly **4 channels**. If not, it prints an error and skips the file.
2.  **Chunking:** Each mono channel is resampled to **16,000 Hz** and sliced into **1-second (1000ms) chunks**, with a **0.5-second (500ms) overlap**.
3.  **Preprocessing:** Each 1-second chunk undergoes:
      * A **250 Hz high-pass filter** to remove low-frequency rumble.
      * Spectrogram calculation (STFT) and conversion to Decibels (dB).
      * **No noise profile subtraction** is performed in this version.
4.  **Sub-Segment Analysis:** The 1-second spectrogram is divided into **10 smaller sub-segments**.
5.  **Detection:** The script analyzes each sub-segment for a specific acoustic signature in the **300-1500 Hz** range:
      * It first checks if the segment's raw amplitude is within a specific range (`y_segment_max > 0.01` and `y_segment_max < 0.12`). This acts as a filter to ignore both silence and very loud non-mosquito sounds.
      * It then searches for a prominent frequency peak followed by a significant dip (a `db_difference` greater than the `threshold` of **12**).
6.  **Thresholding:** A 1-second chunk is classified as a "mosquito sound" **if at least 3 of its 10 sub-segments** trigger the detection logic (`mosquito_count >= 3`).

-----

## Inputs and Outputs

### Inputs

  * **Audio Files (`input_root`):** A folder (e.g., `./input_recordings`) containing **4-channel** `.wav` files.

### Outputs

All outputs are saved to the `output_root` directory (e.g., `./preprocessed_database`).

1.  **Detection CSVs:** For *each* input file processed, a corresponding CSV is created (e.g., `Test0001.csv`).
      * **If detections are found:** The CSV will list the metadata (filename, channel, start\_ms, end\_ms) for *only* the 1-second chunks that were positively identified.
      * **If no detections are found:** The script prints "no mosquito sound found" and no CSV file is created for that input.
2.  **Audio Segments (`output_root`):** The script saves the actual 1-second `.wav` audio chunks that passed the detection.
      * *Example:* If `Test0001.csv` lists a detection for channel 1 at 500ms, the file `./preprocessed_database/Test0001_1_500.wav` will be created.
3.  **Other Segments:** Two other folders are created:
      * `..._not_selected`: For segments that *failed* the check.
      * `..._speech`: For segments identified as speech by VAD.
4.  **(Optional) Saving Negative/Speech Samples:** The script can be modified to save all segments. To save negative (non-mosquito) samples, you must **uncomment** the following line in the `anal_chunks` function:
    ```python
    #chunk.export(not_selected_file, format="wav", parameters=["-ar", "16000", "-ac", "1", "-sample_fmt", "s16"])
    ```
    To save the speech segments that were filtered out, **uncomment** this line in the same function:
    ```python
    #chunk.export(chunk_speech_file, format="wav", parameters=["-ar", "16000", "-ac", "1", "-sample_fmt", "s16"])
    ```
	