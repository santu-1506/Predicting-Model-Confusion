import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_embedding=False):
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor of shape (batch, 3, 32, 32)
            return_embedding: If True, also return the 256-dim hidden layer activation
            
        Returns:
            logits: Class logits of shape (batch, num_classes)
            hidden (optional): 256-dim embedding from second-to-last layer
        """
        # Extract convolutional features
        features_out = self.features(x)
        flat = torch.flatten(features_out, 1)
        
        # Walk through the classifier layers manually to grab the hidden embedding
        # classifier = [Flatten(0), Linear(1), ReLU(2), Dropout(3), Linear(4)]
        hidden = self.classifier[1](flat)   # Linear(2048 -> 256)
        hidden = self.classifier[2](hidden) # ReLU
        logits = self.classifier[4](self.classifier[3](hidden))  # Dropout -> Linear(256 -> 10)
        
        if return_embedding:
            return logits, hidden
        return logits

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    # ONLY train on base_train_loader!
    base_train_loader, calibration_loader, test_loader = get_dataloaders(batch_size=128)
    
    # 2. Setup Model, Loss, Optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("Starting training...")
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in base_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

    # 4. Save Model
    save_path = "models/base_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
