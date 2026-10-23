import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import MultimodalPaddyDataset
from models.cross_attention_net import AgroVisionNet
import config

def train_model():
    train_dataset = MultimodalPaddyDataset(csv_file="data/train.csv", img_dir="data/images", is_train=True)
    val_dataset = MultimodalPaddyDataset(csv_file="data/val.csv", img_dir="data/images", is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = AgroVisionNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    print(f"Starting Training on {config.DEVICE}...")
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, weather, labels in train_loader:
            images, weather, labels = images.to(config.DEVICE), weather.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs, _ = model(images, weather)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / len(train_dataset)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc*100:.2f}%")

    torch.save(model.state_dict(), "agrovision_best.pth")
    print("Model checkpoint saved to agrovision_best.pth")

if __name__ == "__main__":
    train_model()