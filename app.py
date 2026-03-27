import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# ESC-50 class labels (index 0-49)
ESC50_LABELS = [
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea waves", "crackling fire", "crickets", "chirping birds",
    "water drops", "wind", "pouring water", "toilet flush", "thunderstorm",
    "crying baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing teeth", "snoring", "drinking/sipping",
    "door knock", "mouse click", "keyboard typing", "door creak", "can opening",
    "washing machine", "vacuum cleaner", "clock alarm", "clock tick", "glass breaking",
    "helicopter", "chainsaw", "siren", "car horn", "engine",
    "train", "church bells", "airplane", "fireworks", "hand saw",
]


# ── Models ────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        return self.fc(self.global_pool(x).view(x.size(0), -1))


class ResNetAudio(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.resnet.fc.in_features, num_classes))

    def forward(self, x):
        return self.resnet(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple cache so we don't reload the model on every chunk
_model_cache = {}

def get_available_models():
    return sorted(glob.glob("*.pth"))


def load_model(pth_path):
    if pth_path in _model_cache:
        return _model_cache[pth_path]
    name = os.path.basename(pth_path).lower()
    model = ResNetAudio(num_classes=50) if "resnet" in name else SimpleCNN(num_classes=50)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    _model_cache[pth_path] = model
    return model


def infer_from_array(y, sr, model):
    if sr != 22050:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=22050)
        sr = 22050
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probs


def audio_to_tensor(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float().to(device), y, sr, mel_db


def plot_spectrogram(mel_db, sr=22050):
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    return fig


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2))
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, linewidth=0.5, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    plt.tight_layout()
    return fig


# ── Tab 1: Single clip classify ───────────────────────────────────────────────

def predict(audio_path, model_path, top_k):
    if audio_path is None:
        return "No audio provided.", None, None, None
    available = get_available_models()
    if not available:
        return "No .pth model files found. Train a model first.", None, None, None
    if model_path not in available:
        model_path = available[0]
    model = load_model(model_path)
    tensor, y, sr, mel_db = audio_to_tensor(audio_path)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    top_k = int(top_k)
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_labels = {ESC50_LABELS[i]: float(probs[i]) for i in top_indices}
    summary = f"Top prediction: **{ESC50_LABELS[top_indices[0]]}** ({probs[top_indices[0]]*100:.1f}%)"
    return summary, top_labels, plot_spectrogram(mel_db, sr), plot_waveform(y, sr)


# ── Tab 2: Live captioning ────────────────────────────────────────────────────

LIVE_SR = 22050
WINDOW_SECONDS = 5          # sliding window length fed to the model
MIN_SECONDS = 1             # minimum audio before running inference
INFER_EVERY_N_CHUNKS = 8    # run inference every N chunks to avoid overloading


def process_stream(audio_chunk, state):
    """
    Called by Gradio for each mic chunk.
    state = {"buffer": np.array, "chunk_count": int, "history": [str]}
    """
    if audio_chunk is None:
        return state, state["current_label"], "\n".join(state["history"])

    sr, chunk = audio_chunk

    # Convert to mono float32
    if chunk.ndim > 1:
        chunk = chunk.mean(axis=1)
    chunk = chunk.astype(np.float32)
    if chunk.max() > 1.0:
        chunk = chunk / 32768.0

    # Resample to 22050 if browser sends a different rate
    if sr != LIVE_SR:
        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=LIVE_SR)

    # Append to rolling buffer, keep last WINDOW_SECONDS
    buffer = np.concatenate([state["buffer"], chunk])
    max_samples = WINDOW_SECONDS * LIVE_SR
    if len(buffer) > max_samples:
        buffer = buffer[-max_samples:]

    state["buffer"] = buffer
    state["chunk_count"] += 1

    # Only run inference every N chunks and once we have enough audio
    if state["chunk_count"] % INFER_EVERY_N_CHUNKS != 0:
        return state, state["current_label"], "\n".join(state["history"])
    if len(buffer) < MIN_SECONDS * LIVE_SR:
        return state, "Listening...", "\n".join(state["history"])

    available = get_available_models()
    if not available:
        return state, "No model found.", "\n".join(state["history"])

    model = load_model(state["model_path"] if state["model_path"] in available else available[0])
    probs = infer_from_array(buffer.copy(), LIVE_SR, model)
    top_idx = int(np.argmax(probs))
    label = ESC50_LABELS[top_idx]
    conf = probs[top_idx] * 100

    current = f"{label} ({conf:.0f}%)"
    state["current_label"] = current

    # Only log to history when the top label changes
    if label != state["last_logged_label"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        state["history"].insert(0, f"[{timestamp}]  {label}  ({conf:.0f}%)")
        state["history"] = state["history"][:50]  # keep last 50 entries
        state["last_logged_label"] = label

    return state, current, "\n".join(state["history"])


def make_live_state(model_path):
    return {
        "buffer": np.array([], dtype=np.float32),
        "chunk_count": 0,
        "current_label": "Listening...",
        "last_logged_label": "",
        "history": [],
        "model_path": model_path or "",
    }


def clear_live(model_path):
    return make_live_state(model_path), "Listening...", ""


# ── UI ────────────────────────────────────────────────────────────────────────

def refresh_models():
    choices = get_available_models()
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


with gr.Blocks(title="ESC-50 Sound Classifier") as demo:
    gr.Markdown("# ESC-50 Sound Classifier")

    with gr.Tabs():

        # ── Tab 1 ──────────────────────────────────────────────────────────────
        with gr.Tab("Classify Clip"):
            gr.Markdown("Upload a file or record a short clip to classify it.")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(label="Audio Input", type="filepath", sources=["upload", "microphone"])
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Model checkpoint",
                            choices=get_available_models(),
                            value=get_available_models()[0] if get_available_models() else None,
                            interactive=True,
                        )
                        refresh_btn = gr.Button("Refresh", size="sm")
                    top_k_slider = gr.Slider(label="Top-K predictions", minimum=1, maximum=10, step=1, value=5)
                    run_btn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=2):
                    summary_out = gr.Markdown()
                    label_out = gr.Label(label="Prediction probabilities")
            with gr.Row():
                spec_out = gr.Plot(label="Mel Spectrogram")
                wave_out = gr.Plot(label="Waveform")

            refresh_btn.click(fn=refresh_models, outputs=model_dropdown)
            run_btn.click(fn=predict, inputs=[audio_input, model_dropdown, top_k_slider],
                          outputs=[summary_out, label_out, spec_out, wave_out])

        # ── Tab 2 ──────────────────────────────────────────────────────────────
        with gr.Tab("Live Captioning"):
            gr.Markdown(
                "Streams your microphone continuously. "
                "The model classifies a rolling **5-second window** and logs every time the detected sound changes."
            )

            live_model_dropdown = gr.Dropdown(
                label="Model checkpoint",
                choices=get_available_models(),
                value=get_available_models()[0] if get_available_models() else None,
                interactive=True,
            )

            live_state = gr.State(value=make_live_state(
                get_available_models()[0] if get_available_models() else ""
            ))

            mic_input = gr.Audio(
                label="Microphone",
                sources=["microphone"],
                streaming=True,
            )

            with gr.Row():
                current_label = gr.Textbox(label="Current sound", interactive=False, scale=1)
                clear_btn = gr.Button("Clear history", scale=0)

            history_box = gr.Textbox(
                label="Detection history (newest first)",
                lines=12,
                interactive=False,
            )

            # Update state when model selection changes
            live_model_dropdown.change(
                fn=lambda path, state: ({**state, "model_path": path}, state["current_label"], "\n".join(state["history"])),
                inputs=[live_model_dropdown, live_state],
                outputs=[live_state, current_label, history_box],
            )

            mic_input.stream(
                fn=process_stream,
                inputs=[mic_input, live_state],
                outputs=[live_state, current_label, history_box],
            )

            clear_btn.click(
                fn=clear_live,
                inputs=[live_model_dropdown],
                outputs=[live_state, current_label, history_box],
            )

if __name__ == "__main__":
    demo.launch()
