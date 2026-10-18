import os
import json
import asyncio

import numpy as np
import librosa

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from classical import load_models


# ============================================================
# CONFIGURATION
# ============================================================

MODELS_DIR = os.environ.get("MODELS_DIR", "models")

TARGET_SR = 16000
WINDOW_SIZE = 16000       # 1 second
HOP_SIZE = 4000           # 250 ms

COMMAND_COOLDOWN = 1.2


# ============================================================
# LOAD MODELS
# ============================================================

# {type: Recognizer}, mis. "gmm_hmm", "dtw_knn", "gmm", "svm"
MODELS = load_models(MODELS_DIR)

if MODELS:
    print(
        "Models loaded:",
        ", ".join(m.display_name for m in MODELS.values())
    )

    # Warm-up: panggilan pertama librosa (JIT numba) butuh ~2 detik,
    # jadi dilakukan sekali di sini, bukan saat command pertama.
    warmup_audio = np.random.default_rng(0).uniform(
        -0.1, 0.1, TARGET_SR
    ).astype(np.float32)

    for recognizer in MODELS.values():
        recognizer.predict_proba(warmup_audio)
else:
    print(
        f"Warning: no models found in '{MODELS_DIR}'. "
        "Train them in the notebook and copy the .pkl files there."
    )

DEFAULT_MODEL = next(iter(MODELS), None)


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
# MODEL API
# ============================================================

@app.get("/api/models")
async def get_models():

    return [
        {
            "key": key,
            "name": recognizer.display_name
        }
        for key, recognizer in MODELS.items()
    ]


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
# MODEL INFERENCE
# ============================================================

def predict_command(audio, recognizer):

    probabilities = recognizer.predict_proba(
        preprocess_audio(audio)
    )

    predicted_index = int(np.argmax(probabilities))

    command = recognizer.class_names[predicted_index]

    return command, float(probabilities[predicted_index])


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

    # Model dipilih per koneksi (tiap tab browser bisa berbeda)
    model_key = DEFAULT_MODEL

    async def select_model(key):

        nonlocal model_key, last_command

        if key not in MODELS:

            await websocket.send_json({
                "type": "error",
                "message": f"Unknown model: {key}"
            })

            return False

        model_key = key
        last_command = None

        print("Model selected:", key)

        return True

    try:

        while True:

            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            # ------------------------------------------------
            # CONFIG / SET_MODEL MESSAGE
            # ------------------------------------------------

            if "text" in message and message["text"]:

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

                    if data.get("model"):
                        await select_model(data["model"])

                    if model_key is None:

                        await websocket.send_json({
                            "type": "error",
                            "message": "No models available on the server"
                        })

                        continue

                    await websocket.send_json({
                        "type": "status",
                        "message": "Microphone connected",
                        "model": model_key
                    })

                elif data.get("type") == "set_model":

                    if await select_model(data.get("model")):

                        await websocket.send_json({
                            "type": "status",
                            "message": "Model switched",
                            "model": model_key
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
                    model_key is not None
                    and len(audio_buffer) >= WINDOW_SIZE
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

                    recognizer = MODELS[model_key]

                    command, confidence = await asyncio.to_thread(
                        predict_command,
                        current_audio,
                        recognizer
                    )

                    current_time = asyncio.get_event_loop().time()

                    accepted = False

                    if (
                        confidence >= recognizer.confidence_threshold
                        and command != "Unknown"
                    ):

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

                        "model": model_key,

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
