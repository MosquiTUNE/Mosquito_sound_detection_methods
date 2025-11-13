# --- PARAMETERS ---
INPUT_ROOT = "./ABUZZ"
OUTPUT_ROOT = "./ABUZZ_preprocessed_database"

# --- IMPORTS (placed right after the parameters, as requested) ---
import os
from pathlib import Path

import numpy as np
import librosa

from pydub import AudioSegment
from scipy.signal import butter, filtfilt

import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# Ensure output root exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- DSP & DETECTION HELPERS ---
# Notes:
# - All comments are in English; Hungarian comments from the original are translated.
# - Function/variable names are anglicized but behavior is preserved.

SUPPORTED_EXTS = [".wav", ".mp4", ".m4a", ".amr"]
_silero_vad_model = None  # initialized in the RUN cell

def highpass_filter(y: np.ndarray, sr: int, cutoff: float = 100.0, order: int = 4) -> np.ndarray:
    """Apply a Butterworth high-pass filter to a mono signal.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal in float32 range [-1, 1].
    sr : int
        Sample rate in Hz.
    cutoff : float
        High-pass cutoff frequency in Hz (default 100 Hz).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal (same shape as input).
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, y)


def analyze_spectrogram_segment(segment_db: np.ndarray, sr: int, n_fft: int, hop_length: int, threshold_db: float = 15.0) -> bool:
    """Heuristic detector on a spectrogram segment (in dB).

    Strategy
    --------
    1) Aggregate by frequency via max over time.
    2) Focus on 300–1500 Hz (typical mosquito fundamental range).
    3) Find a first local minimum; then a subsequent max; then the next local minimum.
    4) Accept if the dB drop after the max exceeds 'threshold_db' and the peak < 1500 Hz.
    """
    aggregated = np.max(segment_db, axis=1)

    # rFFT frequency axis
    freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # Focus band
    band_idx = np.where((freqs >= 300) & (freqs <= 1500))[0]
    band_spec = aggregated[band_idx]
    if band_spec.size < 3:
        return False

    # First local minimum via peak-pick on inverted curve
    if band_spec[0] >= band_spec[1]:
        mins = librosa.util.peak_pick(-band_spec, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.3, wait=1)
        first_min_global = band_idx[mins[0]] if len(mins) > 0 else band_idx[0]
    else:
        first_min_global = band_idx[0]

    # Maximum after that minimum
    tail_after_first_min = aggregated[first_min_global:]
    max_global = first_min_global + int(np.argmax(tail_after_first_min))

    # Next minimum after the maximum
    tail_after_max = aggregated[max_global:]
    mins_after_max = librosa.util.peak_pick(-tail_after_max, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.3, wait=1)
    post_max_min_global = (
        max_global + int(mins_after_max[0])
        if len(mins_after_max) > 0
        else max_global + int(np.argmin(tail_after_max))
    )

    max_db = float(aggregated[max_global])
    min_db = float(aggregated[post_max_min_global])
    db_drop = max_db - min_db

    # Reject if the main peak is out of our focus band
    if freqs[max_global] > 1500:
        return False

    return db_drop > threshold_db


def filter_audio_chunk(chunk) -> bool:
    """Return True if a ~1 s chunk likely contains mosquito sound.

    Pipeline
    --------
    - Convert pydub samples to float32 in [-1, 1]
    - High-pass at 100 Hz
    - Compute dB spectrogram (n_fft=512, hop_length=50)
    - Slice into 10 equal time segments; count “positives”
    - Accept if >= 3 positive segments
    """
    arr = chunk.get_array_of_samples()
    tcode = arr.typecode  # 'h' (int16) or 'i' (int32)
    y = np.array(arr, dtype=np.float32)

    if tcode == "h":
        y /= (2 ** 15)
    elif tcode == "i":
        y /= (2 ** 31)
    else:
        raise ValueError(f"Unsupported sample type from pydub array: {tcode}")

    sr = int(chunk.frame_rate)

    # High-pass filter
    y = highpass_filter(y, sr, cutoff=100.0, order=4)

    # Spectrogram in dB
    n_fft = 512
    hop_length = 50
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Time segmentation
    num_segments = 10
    spec_cols = S_db.shape[1]
    seg_cols = max(1, spec_cols // num_segments)

    raw_len = len(y)
    raw_seg = max(1, raw_len // num_segments)

    hits = 0
    for i in range(num_segments):
        c0 = i * seg_cols
        c1 = spec_cols if i == num_segments - 1 else c0 + seg_cols
        seg_db = S_db[:, c0:c1]

        r0 = i * raw_seg
        r1 = raw_len if i == num_segments - 1 else r0 + raw_seg
        seg_peak = float(np.max(np.abs(y[r0:r1])))

        # Skip very weak segments (reduces false positives and saves compute)
        if seg_peak > 0.02 and analyze_spectrogram_segment(seg_db, sr, n_fft, hop_length):
            hits += 1

    return hits >= 3


def is_speech(chunk_8k) -> bool:
    """Return True if Silero VAD detects speech in this chunk.

    Notes
    -----
    - Pipeline chunks are 8 kHz mono; Silero works best at 16 kHz,
      so we upsample to 16 kHz before inference.
    - '_silero_vad_model' is initialized in the RUN cell.
    """
    if _silero_vad_model is None:
        raise RuntimeError("Silero VAD model not initialized yet. Run the RUN cell.")

    arr = chunk_8k.get_array_of_samples()
    tcode = arr.typecode
    y = np.array(arr, dtype=np.float32)

    if tcode == "h":
        y /= (2 ** 15)
    elif tcode == "i":
        y /= (2 ** 31)
    else:
        raise ValueError(f"Unsupported sample type from pydub array: {tcode}")

    # 8 kHz -> 16 kHz for Silero
    y16 = librosa.resample(y, orig_sr=8000, target_sr=16000)
    y16_t = torch.tensor(y16, dtype=torch.float32)

    stamps = get_speech_timestamps(y16_t, _silero_vad_model, return_seconds=True)
    return len(stamps) > 0


def split_audio_into_chunks(input_file: str):
    """Load an audio file, force 8 kHz mono, then split into 1.0 s chunks with 0.5 s hop.
    Returns a list of pydub.AudioSegment.
    """
    chunks = []
    try:
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(8000).set_channels(1)

        duration_ms = len(audio)
        window_ms = 1000  # 1.0 s
        step_ms = 500     # 0.5 s hop

        for start_ms in range(0, max(0, duration_ms - window_ms + 1), step_ms):
            end_ms = start_ms + window_ms
            chunks.append(audio[start_ms:end_ms])
    except Exception as e:
        print(f"[split] Failed for {input_file}: {e}")

    return chunks


# Centralized export parameters for pydub .export()
EXPORT_FORMAT = "wav"
EXPORT_PARAMS = ["-ar", "16000", "-ac", "1", "-sample_fmt", "s16"]  # 16 kHz mono s16 for uniformity

def export_wav(chunk: AudioSegment, filepath: str):
    """Centralized export enforcing consistent audio format and parameters."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    chunk.export(filepath, format=EXPORT_FORMAT, parameters=EXPORT_PARAMS)


def save_chunks(chunks, out_dir: str, base_stem: str):
    """Save chunks into three folders:
      - <out_dir>/                : mosquito-positive chunks
      - <out_dir>_speech/         : chunks with detected speech
      - <out_dir>_not_selected/   : negatives
    """
    os.makedirs(out_dir, exist_ok=True)

    speech_dir = os.path.join(os.path.dirname(out_dir), os.path.basename(out_dir) + "_speech")
    neg_dir = os.path.join(os.path.dirname(out_dir), os.path.basename(out_dir) + "_not_selected")
    os.makedirs(speech_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        # 1) Speech exclusion
        if is_speech(chunk):
            f = os.path.join(speech_dir, f"{base_stem}_chunk{idx + 1}.wav")
            export_wav(chunk, f)
            if idx % 50 == 0:
                print(f"[{idx:05d}] saved SPEECH: {f}")
            continue

        # 2) Mosquito heuristic
        if filter_audio_chunk(chunk):
            f = os.path.join(out_dir, f"{base_stem}_chunk{idx + 1}.wav")
            export_wav(chunk, f)
            if idx % 50 == 0:
                print(f"[{idx:05d}] saved MOSQUITO: {f}")
        else:
            f = os.path.join(neg_dir, f"{base_stem}_chunk{idx + 1}.wav")
            export_wav(chunk, f)
            if idx % 50 == 0:
                print(f"[{idx:05d}] saved NEGATIVE: {f}")


def process_directory_recursive(input_root: str, output_root: str):
    """Process only the first-level subfolders of 'input_root'.
    For each subfolder, walk all files recursively and process supported formats.
    """
    for root, dirs, files in os.walk(input_root):
        if root != input_root:
            # Only operate on immediate subdirectories
            continue

        for sub in dirs:
            sub_in = os.path.join(root, sub)
            sub_out = os.path.join(output_root, sub)
            os.makedirs(sub_out, exist_ok=True)

            for sub_root, _, sub_files in os.walk(sub_in):
                for fname in sub_files:
                    if not any(fname.lower().endswith(ext) for ext in SUPPORTED_EXTS):
                        continue

                    in_path = os.path.join(sub_root, fname)
                    base = Path(fname).stem

                    chunks = split_audio_into_chunks(in_path)
                    save_chunks(chunks, sub_out, base)

# --- RUN ---
# Initialize Silero VAD once and store globally
if _silero_vad_model is None:
    _silero_vad_model = load_silero_vad()

# Start processing
process_directory_recursive(INPUT_ROOT, OUTPUT_ROOT)

