import os
import asyncio
from collections import deque

import numpy as np
import torch
import librosa

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from model import DSCNN


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best_dscnn.pth"

TARGET_SR = 16000
WINDOW_SIZE = 16000       # 1 second
HOP_SIZE = 4000           # 250 ms

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

CONFIDENCE_THRESHOLD = 0.80
COMMAND_COOLDOWN = 1.2

CLASS_NAMES = [
    "Play",
    "Pause",
    "Next_track",
    "Previous_track",
    "Unknown"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

model = DSCNN(num_classes=5)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Jika checkpoint merupakan state_dict langsung
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Real-Time Speech Command Recognition"
)


# Static frontend
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Music files
app.mount(
    "/music",
    StaticFiles(directory="music"),
    name="music"
)


# ============================================================
# ROOT PAGE
# ============================================================

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ============================================================
# SONG API
# ============================================================

@app.get("/api/songs")
async def get_songs():

    music_directory = "music"

    if not os.path.exists(music_directory):
        return []

    supported_extensions = (
        ".mp3",
        ".wav",
        ".ogg",
        ".m4a"
    )

    songs = []

    for filename in os.listdir(music_directory):

        if filename.lower().endswith(supported_extensions):

            songs.append({
                "name": os.path.splitext(filename)[0],
                "filename": filename,
                "url": "/music/" + filename
            })

    songs.sort(key=lambda x: x["name"].lower())

    return songs


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio):

    audio = np.asarray(audio, dtype=np.float32)

    # Normalize
    max_val = np.max(np.abs(audio))

    if max_val > 0:
        audio = audio / max_val

    # Pad / trim to exactly 1 second
    if len(audio) < WINDOW_SIZE:

        audio = np.pad(
            audio,
            (0, WINDOW_SIZE - len(audio)),
            mode="constant"
        )

    else:

        audio = audio[:WINDOW_SIZE]

    return audio


# ============================================================
# MFCC
# ============================================================

def extract_mfcc(audio):

    audio = preprocess_audio(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    return mfcc


# ============================================================
# MODEL INFERENCE
# ============================================================

def predict_command(audio):

    mfcc = extract_mfcc(audio)

    # (40, 101)
    mfcc_tensor = torch.tensor(
        mfcc,
        dtype=torch.float32
    )

    # (1, 1, 40, 101)
    mfcc_tensor = mfcc_tensor.unsqueeze(0).unsqueeze(0)

    mfcc_tensor = mfcc_tensor.to(device)

    with torch.no_grad():

        output = model(mfcc_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted.item()
    confidence_value = confidence.item()

    command = CLASS_NAMES[predicted_index]

    return command, confidence_value


# ============================================================
# RESAMPLE
# ============================================================

def resample_audio(audio, original_sr):

    if original_sr == TARGET_SR:
        return audio

    audio = librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=TARGET_SR
    )

    return audio.astype(np.float32)


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    print("WebSocket client connected.")

    audio_buffer = np.zeros(
        0,
        dtype=np.float32
    )

    input_sample_rate = None

    samples_since_prediction = 0

    last_command = None
    last_command_time = 0

    try:

        while True:

            message = await websocket.receive()

            # ------------------------------------------------
            # CONFIG MESSAGE
            # ------------------------------------------------

            if "text" in message and message["text"]:

                import json

                data = json.loads(
                    message["text"]
                )

                if data.get("type") == "config":

                    input_sample_rate = int(
                        data.get(
                            "sample_rate",
                            TARGET_SR
                        )
                    )

                    print(
                        "Input sample rate:",
                        input_sample_rate
                    )

                    await websocket.send_json({
                        "type": "status",
                        "message": "Microphone connected"
                    })

            # ------------------------------------------------
            # AUDIO DATA
            # ------------------------------------------------

            elif "bytes" in message and message["bytes"]:

                audio_chunk = np.frombuffer(
                    message["bytes"],
                    dtype=np.float32
                ).copy()

                if len(audio_chunk) == 0:
                    continue

                if input_sample_rate is None:

                    input_sample_rate = TARGET_SR

                # Resample browser audio → 16 kHz
                audio_chunk = resample_audio(
                    audio_chunk,
                    input_sample_rate
                )

                # Add to buffer
                audio_buffer = np.concatenate(
                    (
                        audio_buffer,
                        audio_chunk
                    )
                )

                # Keep only enough audio
                if len(audio_buffer) > WINDOW_SIZE * 2:

                    audio_buffer = audio_buffer[
                        -WINDOW_SIZE * 2:
                    ]

                samples_since_prediction += len(
                    audio_chunk
                )

                # ------------------------------------------------
                # RUN MODEL EVERY 250 ms
                # ------------------------------------------------

                if (
                    len(audio_buffer) >= WINDOW_SIZE
                    and samples_since_prediction >= HOP_SIZE
                ):

                    samples_since_prediction = 0

                    current_audio = audio_buffer[
                        -WINDOW_SIZE:
                    ]

                    # --------------------------------------------
                    # SIMPLE SILENCE DETECTION
                    # --------------------------------------------

                    rms = np.sqrt(
                        np.mean(
                            current_audio ** 2
                        )
                    )

                    if rms < 0.005:

                        await websocket.send_json({
                            "type": "prediction",
                            "command": None,
                            "confidence": 0,
                            "status": "silence"
                        })

                        continue

                    # --------------------------------------------
                    # MODEL PREDICTION
                    # --------------------------------------------

                    command, confidence = await asyncio.to_thread(
                        predict_command,
                        current_audio
                    )

                    current_time = asyncio.get_event_loop().time()

                    accepted = False

                    if confidence >= CONFIDENCE_THRESHOLD:

                        if (
                            command != last_command
                            or
                            current_time - last_command_time
                            >= COMMAND_COOLDOWN
                        ):

                            accepted = True

                            last_command = command
                            last_command_time = current_time

                    # --------------------------------------------
                    # SEND RESULT
                    # --------------------------------------------

                    await websocket.send_json({

                        "type": "prediction",

                        "command": command,

                        "confidence": round(
                            confidence,
                            4
                        ),

                        "accepted": accepted,

                        "status": "prediction"

                    })

    except WebSocketDisconnect:

        print(
            "WebSocket client disconnected."
        )

    except Exception as e:

        print(
            "WebSocket error:",
            e
        )

        try:

            await websocket.close()

        except:
            pass
