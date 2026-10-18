# ============================================================
# Classical speech recognizers: feature extraction + inference
#
# File ini juga disalin apa adanya ke notebook training
# (Model_Train/Speech_Recognition_project.ipynb). Jika diubah,
# salin ulang ke notebook dan latih ulang model.
#
# Input semua fungsi: audio 16 kHz yang sudah di-normalize dan
# di-pad/trim menjadi 1 detik (preprocess_audio).
# ============================================================

import glob
import os

import joblib
import librosa
import numpy as np


TARGET_SR = 16000

N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

N_MFCC_CLASSICAL = 13
TRIM_TOP_DB = 30
MIN_SPEECH_SAMPLES = int(0.1 * TARGET_SR)   # minimal 100 ms setelah trim
N_SEGMENTS = 3

DEFAULT_CONFIDENCE_THRESHOLD = 0.80

MODEL_ORDER = ["gmm_hmm", "dtw_knn", "gmm", "svm"]


# ============================================================
# Features
# ============================================================

def mfcc_deltas(audio):
    """
    Trim silence, lalu MFCC 13 + delta + delta-delta.

    Output: shape = (39, time_frames), belum dinormalisasi
    """

    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=TRIM_TOP_DB
    )

    if len(trimmed) >= MIN_SPEECH_SAMPLES:
        audio = trimmed

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC_CLASSICAL,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.vstack([mfcc, delta1, delta2])


def frame_features(audio):
    """
    Fitur per frame untuk GMM-HMM, GMM, dan DTW.
    CMVN per utterance (normalisasi mean & variance per koefisien).

    Output: shape = (time_frames, 39)
    """

    features = mfcc_deltas(audio)

    features = (
        features - features.mean(axis=1, keepdims=True)
    ) / (
        features.std(axis=1, keepdims=True) + 1e-8
    )

    return features.T.astype(np.float64)


def summary_features(audio):
    """
    Fitur fixed-length untuk SVM (tanpa CMVN, karena CMVN membuat
    mean menjadi 0). Mean dari 3 segmen waktu menjaga urutan bunyi
    (mis. "next" vs "previous"), ditambah std keseluruhan.

    Output: shape = (39 * N_SEGMENTS + 39,) = (156,)
    """

    features = mfcc_deltas(audio)

    segment_means = [
        segment.mean(axis=1)
        for segment in np.array_split(features, N_SEGMENTS, axis=1)
    ]

    return np.concatenate(
        segment_means + [features.std(axis=1)]
    ).astype(np.float64)


FEATURE_FUNCTIONS = {
    "frame": frame_features,
    "summary": summary_features
}


def softmax(x):

    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()

    exp = np.exp(x)

    return exp / exp.sum()


# ============================================================
# Recognizers
#
# Bundle (disimpan dengan joblib):
#   {
#     "type": "gmm_hmm" | "dtw_knn" | "gmm" | "svm",
#     "display_name": str,
#     "class_names": [...],
#     "config": {...},
#     "model": {...}   # objek hmmlearn/sklearn/numpy, isi per type
#   }
# ============================================================

class Recognizer:

    feature_kind = "frame"

    def __init__(self, bundle):

        self.bundle = bundle
        self.type = bundle["type"]
        self.display_name = bundle["display_name"]
        self.class_names = list(bundle["class_names"])
        self.config = bundle.get("config", {})
        self.model = bundle["model"]

        self.confidence_threshold = self.config.get(
            "confidence_threshold",
            DEFAULT_CONFIDENCE_THRESHOLD
        )

    def features(self, audio):
        return FEATURE_FUNCTIONS[self.feature_kind](audio)

    def proba_from_features(self, features):
        raise NotImplementedError

    def predict_proba(self, audio):
        return self.proba_from_features(self.features(audio))

    def proba_batch(self, feature_list):
        return np.array([
            self.proba_from_features(f) for f in feature_list
        ])


class GmmHmmRecognizer(Recognizer):
    """
    model = {"models": {class_name: hmmlearn GMMHMM}}
    Confidence = softmax atas log-likelihood per frame.
    """

    def proba_from_features(self, features):

        scores = [
            self.model["models"][name].score(features)
            for name in self.class_names
        ]

        return softmax(np.array(scores) / len(features))


class GmmRecognizer(Recognizer):
    """
    model = {"models": {class_name: sklearn GaussianMixture}}
    GaussianMixture.score = rata-rata log-likelihood per frame.
    """

    def proba_from_features(self, features):

        scores = [
            self.model["models"][name].score(features)
            for name in self.class_names
        ]

        return softmax(scores)


def dtw_distance(a, b):
    """
    Jarak DTW (euclidean) antara dua sequence (T, 39),
    dinormalisasi dengan panjang warping path.
    """

    D, wp = librosa.sequence.dtw(
        X=a.T,
        Y=b.T,
        metric="euclidean"
    )

    return D[-1, -1] / len(wp)


class DtwKnnRecognizer(Recognizer):
    """
    model = {
        "templates": [array (T, 39), ...],
        "labels": array (n_templates,) index kelas,
        "k": int
    }
    Confidence = vote kNN berbobot 1 / jarak.
    """

    def distances(self, features):
        return np.array([
            dtw_distance(features, template)
            for template in self.model["templates"]
        ])

    def proba_from_distances(self, distances, k=None):

        k = k or self.model["k"]

        nearest = np.argsort(distances)[:k]

        votes = np.zeros(len(self.class_names))

        for i in nearest:
            votes[self.model["labels"][i]] += 1.0 / (distances[i] + 1e-8)

        return votes / votes.sum()

    def proba_from_features(self, features):
        return self.proba_from_distances(self.distances(features))


class SvmRecognizer(Recognizer):
    """
    model = {"pipeline": sklearn Pipeline(StandardScaler, SVC(probability=True))}
    Dilatih dengan label index kelas (0..n_classes-1).
    """

    feature_kind = "summary"

    def proba_from_features(self, features):

        pipeline = self.model["pipeline"]

        proba = pipeline.predict_proba(features[None, :])[0]

        result = np.zeros(len(self.class_names))
        result[pipeline.classes_] = proba

        return result

    def proba_batch(self, feature_list):

        pipeline = self.model["pipeline"]

        proba = pipeline.predict_proba(np.array(feature_list))

        result = np.zeros((len(feature_list), len(self.class_names)))
        result[:, pipeline.classes_] = proba

        return result


RECOGNIZERS = {
    "gmm_hmm": GmmHmmRecognizer,
    "dtw_knn": DtwKnnRecognizer,
    "gmm": GmmRecognizer,
    "svm": SvmRecognizer
}


def build_recognizer(bundle):
    return RECOGNIZERS[bundle["type"]](bundle)


def library_versions():

    import hmmlearn
    import sklearn

    return {
        "numpy": np.__version__,
        "librosa": librosa.__version__,
        "scikit-learn": sklearn.__version__,
        "hmmlearn": hmmlearn.__version__
    }


def load_models(models_dir):
    """
    Load semua bundle *.pkl di models_dir.

    Output: {type: Recognizer}, urut sesuai MODEL_ORDER
    """

    recognizers = {}

    current_versions = library_versions()

    for path in glob.glob(os.path.join(models_dir, "*.pkl")):

        try:
            bundle = joblib.load(path)
            recognizer = build_recognizer(bundle)

        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        trained_versions = recognizer.config.get("versions", {})

        for lib, version in trained_versions.items():
            if current_versions.get(lib) != version:
                print(
                    f"Warning: {os.path.basename(path)} trained with "
                    f"{lib} {version}, running {current_versions.get(lib)}"
                )

        recognizers[recognizer.type] = recognizer

    return {
        key: recognizers[key]
        for key in sorted(
            recognizers,
            key=lambda k: MODEL_ORDER.index(k) if k in MODEL_ORDER else len(MODEL_ORDER)
        )
    }
