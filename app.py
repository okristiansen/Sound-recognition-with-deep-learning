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


def get_available_models():
    return sorted(glob.glob("*.pth"))


def load_model(pth_path):
    name = os.path.basename(pth_path).lower()
    if "resnet" in name:
        model = ResNetAudio(num_classes=50)
    else:
        model = SimpleCNN(num_classes=50)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model


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


# ── Main predict function ─────────────────────────────────────────────────────

def predict(audio_path, model_path, top_k):
    if audio_path is None:
        return "No audio provided.", None, None, None

    available = get_available_models()
    if not available:
        return "No .pth model files found in the project folder. Train a model first.", None, None, None

    if model_path not in available:
        model_path = available[0]

    model = load_model(model_path)
    tensor, y, sr, mel_db = audio_to_tensor(audio_path)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_k = int(top_k)
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_labels = {ESC50_LABELS[i]: float(probs[i]) for i in top_indices}

    spec_fig = plot_spectrogram(mel_db, sr)
    wave_fig = plot_waveform(y, sr)

    summary = f"Top prediction: **{ESC50_LABELS[top_indices[0]]}** ({probs[top_indices[0]]*100:.1f}%)"
    return summary, top_labels, spec_fig, wave_fig


# ── UI ────────────────────────────────────────────────────────────────────────

def refresh_models():
    choices = get_available_models()
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


with gr.Blocks(title="ESC-50 Sound Classifier") as demo:
    gr.Markdown("# ESC-50 Sound Classifier\nUpload or record an audio clip and classify it using a trained model.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Audio Input", type="filepath", sources=["upload", "microphone"])

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Model checkpoint",
                    choices=get_available_models(),
                    value=get_available_models()[0] if get_available_models() else None,
                    interactive=True
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
    run_btn.click(
        fn=predict,
        inputs=[audio_input, model_dropdown, top_k_slider],
        outputs=[summary_out, label_out, spec_out, wave_out]
    )

if __name__ == "__main__":
    demo.launch()
