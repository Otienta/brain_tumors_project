import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, test_loader, lr, weight_decay, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def train(self, save_path='model.pth'):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as t:
                for images, labels in t:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    running_loss += loss.item()
                    t.set_postfix(loss=loss.item())
            print(f"Epoch {epoch+1}, Average Loss: {running_loss/len(self.train_loader)}")
        torch.save(self.model.state_dict(), save_path)
        print(f"Modèle sauvegardé sous {save_path}")

    def evaluate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        accuracy = 100.0 * total_correct / total_samples
        print(f"Test Accuracy: {accuracy:.2f}%")