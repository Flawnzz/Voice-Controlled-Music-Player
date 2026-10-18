import numpy as np
import librosa


# ============================================================
# Configuration
# ============================================================

TARGET_SR = 16000
DURATION = 1.0
TARGET_LENGTH = int(TARGET_SR * DURATION)

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400


# ============================================================
# Audio preprocessing
# ============================================================

def preprocess_audio(file_path):
    """
    Load audio dan melakukan preprocessing yang sama
    dengan proses training di Google Colab.

    Output:
        numpy array dengan shape (16000,)
    """

    # Load audio
    # Resample langsung ke 16 kHz dan convert ke mono
    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    # Normalize amplitude
    max_val = np.max(np.abs(audio))

    if max_val > 0:
        audio = audio / max_val

    # Pad / Trim menjadi 1 detik
    if len(audio) < TARGET_LENGTH:

        pad_length = TARGET_LENGTH - len(audio)

        audio = np.pad(
            audio,
            (0, pad_length),
            mode="constant"
        )

    else:

        audio = audio[:TARGET_LENGTH]

    return audio


# ============================================================
# MFCC extraction
# ============================================================

def extract_mfcc(file_path):
    """
    Preprocessing audio + ekstraksi MFCC.

    Output:
        numpy array dengan shape (40, 101)
    """

    audio = preprocess_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    return mfcc