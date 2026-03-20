from dataset import ESC50Dataset2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import csv
from datetime import datetime


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
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def train_and_evaluate(model, train_loader, test_loader, num_epochs, lr, model_name, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    model = model.to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{model_name}_{timestamp}_log.csv"
    best_model_path = f"{model_name}_{timestamp}_best.pth"

    best_accuracy = 0.0
    epochs_without_improvement = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy"])

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
            avg_loss = running_loss / len(train_loader)
            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            writer.writerow([epoch + 1, f"{avg_loss:.4f}", f"{accuracy:.2f}"])
            scheduler.step()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"[{model_name}] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

    print(f"[{model_name}] Best accuracy: {best_accuracy:.2f}% — saved to {best_model_path}")
    print(f"[{model_name}] Training log saved to {log_path}")
    return best_accuracy


def run_cross_validation(model_class, model_kwargs, num_epochs, lr, model_name, patience=7):
    all_folds = [1, 2, 3, 4, 5]
    fold_accuracies = []

    for test_fold in all_folds:
        train_folds = [f for f in all_folds if f != test_fold]
        print(f"\n{model_name} | Fold {test_fold} as test")

        train_dataset = ESC50Dataset2(
            csv_path="data/meta/esc50.csv",
            audio_dir="data/audio",
            folds=train_folds,
            augment=True
        )
        test_dataset = ESC50Dataset2(
            csv_path="data/meta/esc50.csv",
            audio_dir="data/audio",
            folds=[test_fold],
            augment=False
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = model_class(**model_kwargs)
        fold_name = f"{model_name}_fold{test_fold}"
        acc = train_and_evaluate(model, train_loader, test_loader, num_epochs, lr, fold_name, patience)
        fold_accuracies.append(acc)

    avg = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\n[{model_name}] Cross-validation complete.")
    print(f"[{model_name}] Per-fold accuracies: {[f'{a:.2f}%' for a in fold_accuracies]}")
    print(f"[{model_name}] Mean accuracy: {avg:.2f}%")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nTraining SimpleCNN with 5-fold cross-validation")
    run_cross_validation(SimpleCNN, {"num_classes": 50}, num_epochs=40, lr=0.001, model_name="SimpleCNN")

    # print("\nTraining ResNetAudio with 5-fold cross-validation")
    # run_cross_validation(ResNetAudio, {"num_classes": 50}, num_epochs=40, lr=0.0001, model_name="ResNet")
