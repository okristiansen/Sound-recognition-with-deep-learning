from ml_project.models.dataset2 import ESC50Dataset2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models



# def audio_to_melspectrogram(file_path, sr=22050, n_mels=128, hop_length=512, n_fft=2048):
#     audio, _ = librosa.load(file_path, sr=sr)
#     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     return mel_spec_db

# def visualize_spectogram():
#     spec = audio_to_melspectrogram("data/1-137-A-32.wav")
#     librosa.display.specshow(spec, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title("Mel Spectrogram")
#     plt.show()

# if __name__ == "__main__":
#     visualize_spectogram()


train_dataset = ESC50Dataset2(csv_path="meta/esc50.csv", audio_dir="data", folds=[1, 2, 3, 4], augment=True)
test_dataset = ESC50Dataset2(csv_path="meta/esc50.csv", audio_dir="data", folds=[5], augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNetAudio(nn.Module):
    def __init__(self, num_classes=50):
        super(ResNetAudio, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def train_and_evaluate(model, train_loader, test_loader, num_epochs, lr, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for spectrograms, labels in train_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for spectrograms, labels in test_loader:
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)
                outputs = model(spectrograms)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
        scheduler.step()

    torch.save(model.state_dict(), f"{model_name}.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training SimpleCNN")
simple_model = SimpleCNN(num_classes=50)
train_and_evaluate(simple_model, train_loader, test_loader, num_epochs=40, lr=0.001, model_name="SimpleCNN")

#print("\nTraining ResNet")
#resnet_model = ResNetAudio(num_classes=50)
#train_and_evaluate(resnet_model, train_loader, test_loader, num_epochs=40, lr=0.0001, model_name="ResNet")


