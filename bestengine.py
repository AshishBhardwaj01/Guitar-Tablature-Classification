import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.models as models
import os
import time

class GuitarTabNet(nn.Module):
    def __init__(self, input_channels=3, num_frets=19):
        super(GuitarTabNet, self).__init__()

        # Load Pretrained ResNet18 and modify first conv layer to accept RGB images
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 256) 
        
        # Separate fully connected layers for each string
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, num_frets)
            ) for _ in range(6)
        ])

    def forward(self, x):
        # Feature extraction with ResNet
        features = self.resnet(x)  
        
        # Apply each string branch
        outputs = [branch(features) for branch in self.branches]
        return outputs


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Fixed label smoothing loss function
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        
        # Create smoothed label distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        
        # Make sure target indices are valid
        if target.max() >= self.cls:
            print(f"Warning: Target max value {target.max()} exceeds class count {self.cls}")
            target = torch.clamp(target, 0, self.cls-1)
            
        # Set the correct class probability
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Calculate the KL divergence
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def train_model(model, train_loader, val_loader, epochs=30, device='cuda', lr=0.001):
    # Use Adam optimizer with reduced learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Use ReduceLROnPlateau scheduler instead of cosine - more stable
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Loss function with reduced smoothing
    criterion = LabelSmoothingLoss(classes=19, smoothing=0.05)
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    string_accuracies = [[] for _ in range(6)]
    best_val_loss = float('inf')
    patience = 7  # for early stopping
    counter = 0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = 0
            valid_strings = 0
            
            # Handle labels and calculate loss for each string
            for i, output in enumerate(outputs):
                try:
                    # Get target for this string (shape checking)
                    if labels.dim() > 1 and labels.shape[1] > 1:
                        if labels.shape[1] == 6:  # Each row is a guitar string
                            target = labels[:, i]
                        else:  # One-hot encoded
                            target = torch.argmax(labels[:, i], dim=1) if labels.shape[2] > 1 else labels[:, i]
                    else:
                        # Handle unexpected label shape
                        print(f"Unexpected label shape: {labels.shape}")
                        continue
                    
                    # Skip calculating loss if any target value is invalid
                    if target.max() >= output.shape[1]:
                        print(f"String {i+1} has invalid target max {target.max()}, classes: {output.shape[1]}")
                        continue
                        
                    # Calculate loss for this string
                    string_loss = criterion(output, target)
                    
                    # Check for NaN or Inf values
                    if torch.isfinite(string_loss).all():
                        loss += string_loss
                        valid_strings += 1
                    else:
                        print(f"Warning: Non-finite loss for string {i+1}")
                        
                except Exception as e:
                    print(f"Error in loss calculation for string {i+1}: {e}")
            
            # If we have valid outputs, perform backprop
            if valid_strings > 0:
                # Average loss over valid strings
                loss = loss / valid_strings
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
        
        # Calculate average loss per batch
        avg_train_loss = total_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss, accuracies = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        for i, acc in enumerate(accuracies):
            string_accuracies[i].append(acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Time: {epoch_time:.2f}s, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log individual string accuracies
        for i, acc in enumerate(accuracies):
            print(f"  String {i+1} accuracy: {acc:.2f}%")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'accuracies': accuracies
            }, 'best_guitar_tab_model.pt')
            print(f"Model saved with validation loss: {val_loss:.4f}")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model for final evaluation
    checkpoint = torch.load('best_guitar_tab_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['epoch'], checkpoint['accuracies']


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    batch_count = 0
    
    # Track correct/total predictions for each string
    correct = [0] * 6
    total = [0] * 6
    all_preds = [[] for _ in range(6)]
    all_targets = [[] for _ in range(6)]
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Process each string
            for i, output in enumerate(outputs):
                # Extract target for this string
                if labels.dim() > 1 and labels.shape[1] > 1:
                    if labels.shape[1] == 6:  # Each row is a string
                        target = labels[:, i]
                    else:  # One-hot encoded
                        target = torch.argmax(labels[:, i], dim=1) if labels.shape[2] > 1 else labels[:, i]
                else:
                    # Skip if labels have unexpected shape
                    print(f"Unexpected validation label shape: {labels.shape}")
                    continue
                
                # Skip calculating metrics if any target value is invalid
                if target.max() >= output.shape[1]:
                    print(f"Validation: String {i+1} has invalid target {target.max()}, classes: {output.shape[1]}")
                    continue
                
                # Calculate predictions
                _, predicted = torch.max(output, 1)
                
                # Store predictions and targets for confusion matrix
                all_preds[i].extend(predicted.cpu().numpy())
                all_targets[i].extend(target.cpu().numpy())
                
                # Update accuracy metrics
                correct[i] += (predicted == target).sum().item()
                total[i] += target.size(0)
                
                # Calculate loss
                try:
                    string_loss = criterion(output, target)
                    if torch.isfinite(string_loss).all():
                        total_loss += string_loss.item() / 6  # Average across strings
                        batch_count += 1 / 6  # Count partial batch
                except Exception as e:
                    print(f"Error in validation loss: {e}")
    
    # Calculate average loss
    avg_loss = total_loss / max(batch_count, 1)
    
    # Calculate accuracies
    accuracies = []
    for i in range(6):
        if total[i] > 0:
            accuracy = 100 * correct[i] / total[i]
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
    
    return avg_loss, accuracies


def plot_training_metrics(train_losses, val_losses, string_accuracies):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    for i, accs in enumerate(string_accuracies):
        plt.plot(accs, label=f'String {i+1}')
    plt.title('Validation Accuracy by String')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


def test_model(model, test_loader, device):
    model.eval()
    correct = [0] * 6
    total = [0] * 6
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Process each string
            for i, output in enumerate(outputs):
                try:
                    # Extract target for this string
                    if labels.dim() > 1 and labels.shape[1] > 1:
                        if labels.shape[1] == 6:  # Each row is a string
                            target = labels[:, i]
                        else:  # One-hot encoded
                            target = torch.argmax(labels[:, i], dim=1) if labels.shape[2] > 1 else labels[:, i]
                    else:
                        # Skip if labels have unexpected shape
                        continue
                    
                    # Skip if any target value is invalid
                    if target.max() >= output.shape[1]:
                        continue
                        
                    # Calculate predictions
                    _, predicted = torch.max(output, 1)
                    
                    # Update accuracy metrics
                    correct[i] += (predicted == target).sum().item()
                    total[i] += target.size(0)
                except Exception as e:
                    print(f"Error in testing string {i+1}: {e}")
    
    # Report accuracy
    print("\nTest Results:")
    overall_correct = sum(correct)
    overall_total = sum(total)
    overall_accuracy = 100 * overall_correct / overall_total if overall_total > 0 else 0
    
    for i in range(6):
        accuracy = 100 * correct[i] / total[i] if total[i] > 0 else 0
        print(f"String {i+1} accuracy: {accuracy:.2f}% ({correct[i]}/{total[i]})")
    
    print(f"Overall accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")


def main():
    import time
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    set_seed(42)
    
    # Create model and move to device
    model = GuitarTabNet(input_channels=3, num_frets=19)
    model = model.to(device)
    
    # Load your data
    audio_dir = '/content/Guitar-Tablature-Classification/cqt_images'
    annotation_dir = '/content/Guitar-Tablature-Classification/tablature_segments'
    
    print("Loading data...")
    from my_dataloader import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        batch_size=32,  # Reduced batch size for stability
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    print("Starting training...")
    trained_model, best_epoch, final_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        device=device,
        lr=0.0005  # Lower learning rate
    )
    
    print(f"Best model found at epoch {best_epoch} with accuracies: {final_accuracies}")
    
    # Testing
    print("Testing the trained model...")
    test_model(trained_model, test_loader, device)

if __name__ == "__main__":
    main()
