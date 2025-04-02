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
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

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


# def train_model(model, train_loader, val_loader, epochs=20, device='cuda', lr=0.001):
#     # Use Adam optimizer with reduced learning rate
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
#     # Use ReduceLROnPlateau scheduler instead of cosine - more stable
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
#     # Loss function with reduced smoothing
#     criterion = LabelSmoothingLoss(classes=19, smoothing=0.05)
    
#     # For tracking metrics
#     train_losses = []
#     val_losses = []
#     string_accuracies = [[] for _ in range(6)]
#     best_val_loss = float('inf')
#     patience = 7  # for early stopping
#     counter = 0
    
#     print(f"Starting training on {device}...")
    
#     for epoch in range(epochs):
#         start_time = time.time()
        
#         # Training phase
#         model.train()
#         total_loss = 0
#         batch_count = 0
        
#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs = model(inputs)
            
#             # Calculate loss
#             loss = 0
#             valid_strings = 0
            
#             # Handle labels and calculate loss for each string
#             for i, output in enumerate(outputs):
#                 try:
#                     # Get target for this string (shape checking)
#                     if labels.dim() > 1 and labels.shape[1] > 1:
#                         if labels.shape[1] == 6:  # Each row is a guitar string
#                             target = labels[:, i]
#                         else:  # One-hot encoded
#                             target = torch.argmax(labels[:, i], dim=1) if labels.shape[2] > 1 else labels[:, i]
#                     else:
#                         # Handle unexpected label shape
#                         print(f"Unexpected label shape: {labels.shape}")
#                         continue
                    
#                     # Skip calculating loss if any target value is invalid
#                     if target.max() >= output.shape[1]:
#                         print(f"String {i+1} has invalid target max {target.max()}, classes: {output.shape[1]}")
#                         continue
                        
#                     # Calculate loss for this string
#                     string_loss = criterion(output, target)
                    
#                     # Check for NaN or Inf values
#                     if torch.isfinite(string_loss).all():
#                         loss += string_loss
#                         valid_strings += 1
#                     else:
#                         print(f"Warning: Non-finite loss for string {i+1}")
                        
#                 except Exception as e:
#                     print(f"Error in loss calculation for string {i+1}: {e}")
            
#             # If we have valid outputs, perform backprop
#             if valid_strings > 0:
#                 # Average loss over valid strings
#                 loss = loss / valid_strings
                
#                 # Backward pass with gradient clipping
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 batch_count += 1
        
#         # Calculate average loss per batch
#         avg_train_loss = total_loss / max(batch_count, 1)
#         train_losses.append(avg_train_loss)
        
#         # Validation phase
#         val_loss, accuracies = validate_model(model, val_loader, criterion, device)
#         val_losses.append(val_loss)
        
#         for i, acc in enumerate(accuracies):
#             string_accuracies[i].append(acc)
        
#         # Update learning rate based on validation loss
#         scheduler.step(val_loss)
        
#         # Print epoch results
#         epoch_time = time.time() - start_time
#         print(f"Epoch [{epoch + 1}/{epochs}], "
#               f"Train Loss: {avg_train_loss:.4f}, "
#               f"Val Loss: {val_loss:.4f}, "
#               f"Time: {epoch_time:.2f}s, "
#               f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
#         # Log individual string accuracies
#         for i, acc in enumerate(accuracies):
#             print(f"  String {i+1} accuracy: {acc:.2f}%")
        
#         # Save model if it's the best so far
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'val_loss': val_loss,
#                 'accuracies': accuracies
#             }, 'best_guitar_tab_model.pt')
#             print(f"Model saved with validation loss: {val_loss:.4f}")
#             counter = 0  # Reset early stopping counter
#         else:
#             counter += 1
            
#         # Early stopping
#         if counter >= patience:
#             print(f"Early stopping triggered after {epoch+1} epochs")
#             break
    
#     # Load the best model for final evaluation
#     checkpoint = torch.load('best_guitar_tab_model.pt')
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     return model, checkpoint['epoch'], checkpoint['accuracies']


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


# def main():
#     import time
    
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Set seeds for reproducibility
#     set_seed(42)
    
#     # Create model and move to device
#     model = GuitarTabNet(input_channels=3, num_frets=19).to(device)
#     # Enable multi-GPU if available
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs")
#         model = nn.DataParallel(model)

#     model = model.to(device)
    
#     # Load your data
#     audio_dir = "/kaggle/working/Guitar-Tablature-Classification/cqt_images"
#     annotation_dir = '/kaggle/working/Guitar-Tablature-Classification/tablature_segments'
    
#     print("Loading data...")
#     from my_dataloader import create_dataloaders
#     train_loader, val_loader, test_loader = create_dataloaders(
#         audio_dir=audio_dir,
#         annotation_dir=annotation_dir,
#         batch_size=32,  # Reduced batch size for stability
#         train_ratio=0.8,
#         val_ratio=0.1,
#     )
    
#     print("Starting training...")
#     trained_model, best_epoch, final_accuracies = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         epochs=20,
#         device=device,
#         lr=0.0005  # Lower learning rate
#     )
    
#     print(f"Best model found at epoch {best_epoch} with accuracies: {final_accuracies}")
    
#     # Testing
#     print("Testing the trained model...")
#     test_model(trained_model, test_loader, device)

# if __name__ == "__main__":
#     main()

def visualize_sample_images(dataloader, num_samples=8):
    """
    Visualize a batch of sample images from the dataloader to check for distortion.
    """
    # Get a batch of images
    images, labels = next(iter(dataloader))
    
    # Select random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    sample_images = images[indices]
    sample_labels = labels[indices]
    
    # Create a grid of images
    grid = make_grid(sample_images, nrow=4, normalize=True)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    # Plot the grid
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.title("Sample CQT Images")
    plt.axis('off')
    
    # Print labels for each image
    for i, idx in enumerate(indices):
        label_text = ', '.join([f"S{j+1}:{l.item()}" for j, l in enumerate(labels[idx])])
        plt.text(
            (i % 4) * sample_images.size(3) + sample_images.size(3) // 2,
            (i // 4) * sample_images.size(2) + sample_images.size(2) + 15,
            f"Labels: {label_text}",
            ha='center'
        )
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()
    
    print(f"Sample images visualization saved to 'sample_images.png'")
    
    # Display image stats
    print(f"Image stats - Min: {images.min().item():.2f}, Max: {images.max().item():.2f}, Mean: {images.mean().item():.2f}")
    print(f"Image shape: {images[0].shape}")


def visualize_predictions(model, dataloader, device, num_samples=8):
    """
    Visualize model predictions alongside ground truth labels.
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    
    # Select random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    sample_images = images[indices].to(device)
    sample_labels = labels[indices].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(sample_images)
        predictions = [torch.argmax(output, dim=1) for output in outputs]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    # Plot each sample
    for i, idx in enumerate(indices):
        img = sample_images[i].cpu().permute(1, 2, 0)
        
        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min())
        
        # Plot the image
        axes[i].imshow(img)
        
        # Get ground truth and predictions
        truth = [label.item() for label in sample_labels[i]]
        preds = [pred[i].item() for pred in predictions]
        
        # Create label text
        label_text = ""
        for j in range(6):
            if truth[j] == preds[j]:
                label_text += f"String {j+1}: {truth[j]} (✓)\n"
            else:
                label_text += f"String {j+1}: True: {truth[j]}, Pred: {preds[j]} (✗)\n"
        
        axes[i].set_title(f"Sample {i+1}")
        axes[i].text(1.05, 0.5, label_text, 
                     transform=axes[i].transAxes, 
                     verticalalignment='center',
                     fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.close()
    
    print(f"Prediction visualization saved to 'prediction_visualization.png'")


def visualize_correct_incorrect_distribution(model, dataloader, device):
    """
    Visualize the distribution of correct and incorrect predictions for each string.
    """
    model.eval()
    
    # Track correct/incorrect predictions for each string
    correct = [0] * 6
    incorrect = [0] * 6
    
    with torch.no_grad():
        for inputs, labels in dataloader:
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
                    continue
                
                # Skip if any target value is invalid
                if target.max() >= output.shape[1]:
                    continue
                    
                # Calculate predictions
                _, predicted = torch.max(output, 1)
                
                # Count correct/incorrect
                is_correct = (predicted == target)
                correct[i] += is_correct.sum().item()
                incorrect[i] += (~is_correct).sum().item()
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(6)  # String numbers
    width = 0.35
    
    plt.bar(x - width/2, correct, width, label='Correct')
    plt.bar(x + width/2, incorrect, width, label='Incorrect')
    
    plt.xlabel('Guitar String')
    plt.ylabel('Number of Samples')
    plt.title('Correct vs. Incorrect Predictions by String')
    plt.xticks(x, [f'String {i+1}' for i in range(6)])
    
    # Add accuracy percentage
    for i in range(6):
        total = correct[i] + incorrect[i]
        if total > 0:
            accuracy = correct[i] / total * 100
            plt.text(i, correct[i] + 5, f'{accuracy:.1f}%', ha='center')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_distribution.png')
    plt.close()
    
    print(f"Prediction distribution visualization saved to 'prediction_distribution.png'")


def visualize_confusion_matrices(model, dataloader, device):
    """
    Create confusion matrices for each string.
    """
    model.eval()
    
    # Track predictions and targets for each string
    all_preds = [[] for _ in range(6)]
    all_targets = [[] for _ in range(6)]
    
    with torch.no_grad():
        for inputs, labels in dataloader:
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
                    continue
                
                # Skip if any target value is invalid
                if target.max() >= output.shape[1]:
                    continue
                    
                # Calculate predictions
                _, predicted = torch.max(output, 1)
                
                # Collect predictions and targets
                all_preds[i].extend(predicted.cpu().numpy())
                all_targets[i].extend(target.cpu().numpy())
    
    # Create confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i in range(6):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Skip if not enough data
        if len(all_preds[i]) == 0 or len(all_targets[i]) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
            continue
        
        # Get confusion matrix
        cm = confusion_matrix(all_targets[i], all_preds[i])
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', ax=ax)
        ax.set_title(f'String {i+1} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # Simplify labels if there are many classes
        if cm.shape[0] > 10:
            step = max(1, cm.shape[0] // 10)
            ax.set_xticks(np.arange(0, cm.shape[0], step) + 0.5)
            ax.set_yticks(np.arange(0, cm.shape[0], step) + 0.5)
            ax.set_xticklabels(np.arange(0, cm.shape[0], step))
            ax.set_yticklabels(np.arange(0, cm.shape[0], step))
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    
    print(f"Confusion matrices saved to 'confusion_matrices.png'")


def visualize_model_architecture(model):
    """
    Create a visualization of the model architecture.
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get layers and parameters
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.MaxPool2d)):
            layers.append((name, module.__class__.__name__, count_parameters(module)))
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot as a horizontal bar chart
    layer_names = [f"{layer[0]}" for layer in layers]
    layer_params = [layer[2] for layer in layers]
    
    # Shorten very long layer names
    layer_names = [name[:30] + '...' if len(name) > 30 else name for name in layer_names]
    
    y_pos = np.arange(len(layer_names))
    plt.barh(y_pos, layer_params)
    plt.yticks(y_pos, layer_names)
    plt.xlabel('Number of Parameters')
    plt.title('Model Architecture')
    
    # Add total parameters
    plt.figtext(0.5, 0.01, f'Total trainable parameters: {count_parameters(model):,}', 
                ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('model_architecture.png')
    plt.close()
    
    print(f"Model architecture visualization saved to 'model_architecture.png'")


def visualize_per_fret_accuracy(model, dataloader, device):
    """
    Visualize accuracy per fret position for each string.
    """
    model.eval()
    
    # Track correct/total predictions for each fret position
    num_frets = 19  # Assuming this is the number of frets in your model
    correct_by_fret = [{i: 0 for i in range(num_frets)} for _ in range(6)]
    total_by_fret = [{i: 0 for i in range(num_frets)} for _ in range(6)]
    
    with torch.no_grad():
        for inputs, labels in dataloader:
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
                    continue
                
                # Skip if any target value is invalid
                if target.max() >= output.shape[1]:
                    continue
                    
                # Calculate predictions
                _, predicted = torch.max(output, 1)
                
                # Update counts per fret
                for j in range(len(target)):
                    fret = target[j].item()
                    if fret < num_frets:
                        total_by_fret[i][fret] += 1
                        if predicted[j] == target[j]:
                            correct_by_fret[i][fret] += 1
    
    # Create heatmap
    accuracies = np.zeros((6, num_frets))
    sample_counts = np.zeros((6, num_frets))
    
    for i in range(6):
        for j in range(num_frets):
            total = total_by_fret[i][j]
            sample_counts[i, j] = total
            if total > 0:
                accuracies[i, j] = correct_by_fret[i][j] / total * 100
            else:
                accuracies[i, j] = np.nan  # No samples for this combination
    
    plt.figure(figsize=(15, 8))
    
    # Plot heatmap
    ax = sns.heatmap(accuracies, annot=True, fmt='.1f', cmap='viridis', 
                     vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'})
    
    # Add text showing number of samples
    for i in range(6):
        for j in range(num_frets):
            if not np.isnan(accuracies[i, j]):
                ax.text(j + 0.5, i + 0.85, f"n={int(sample_counts[i, j])}", 
                        ha='center', va='center', fontsize=8, color='black')
    
    plt.title('Accuracy (%) per String and Fret Position')
    plt.xlabel('Fret Position')
    plt.ylabel('String Number')
    plt.yticks(np.arange(6) + 0.5, [f'String {i+1}' for i in range(6)])
    plt.xticks(np.arange(0, num_frets, 2) + 0.5, np.arange(0, num_frets, 2))
    
    plt.tight_layout()
    plt.savefig('fret_accuracy.png')
    plt.close()
    
    print(f"Fret accuracy visualization saved to 'fret_accuracy.png'")


def visualize_loss_curves(train_losses, val_losses, string_accuracies, lr_history=None):
    """
    Enhanced visualization of training metrics.
    """
    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Plot losses
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot string accuracies
    ax2 = fig.add_subplot(3, 1, 2)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    for i, accs in enumerate(string_accuracies):
        ax2.plot(accs, label=f'String {i+1}', color=colors[i], linewidth=2)
    ax2.set_title('Validation Accuracy by String', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate if available
    if lr_history:
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(lr_history, color='purple', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14)
        ax3.set_xlabel('Epochs', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_yscale('log')
        ax3.grid(True, linestyle='--', alpha=0.7)
    else:
        # Plot overall accuracy
        ax3 = fig.add_subplot(3, 1, 3)
        overall_acc = np.mean(string_accuracies, axis=0)
        ax3.plot(overall_acc, color='black', linewidth=2)
        ax3.set_title('Overall Validation Accuracy', fontsize=14)
        ax3.set_xlabel('Epochs', fontsize=12)
        ax3.set_ylabel('Accuracy (%)', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_metrics_detailed.png')
    plt.close()
    
    print(f"Detailed training metrics visualization saved to 'training_metrics_detailed.png'")


# Modified functions to include the visualizations

def train_model(model, train_loader, val_loader, epochs=20, device='cuda', lr=0.001):
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
    lr_history = []
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
        lr_history.append(optimizer.param_groups[0]['lr'])
        
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
        
        # Visualize training metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            visualize_loss_curves(train_losses, val_losses, string_accuracies, lr_history)
    
    # Load the best model for final evaluation
    checkpoint = torch.load('best_guitar_tab_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final detailed visualization
    visualize_loss_curves(train_losses, val_losses, string_accuracies, lr_history)
    
    return model, checkpoint['epoch'], checkpoint['accuracies'], (train_losses, val_losses, string_accuracies)


def main():
    import time
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    set_seed(42)
    
    # Create model and move to device
    model = GuitarTabNet(input_channels=3, num_frets=19).to(device)
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Load your data
    audio_dir = "/kaggle/working/Guitar-Tablature-Classification/cqt_images"
    annotation_dir = '/kaggle/working/Guitar-Tablature-Classification/tablatures'
    
    print("Loading data...")
    from my_dataloader import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        batch_size=32,  # Reduced batch size for stability
        train_ratio=0.8,
        val_ratio=0.1,
    )
    
    # Visualize model architecture
    print("Visualizing model architecture...")
    visualize_model_architecture(model)
    
    # Visualize sample images
    print("Visualizing sample images...")
    visualize_sample_images(train_loader)
    
    print("Starting training...")
    trained_model, best_epoch, final_accuracies, training_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        device=device,
        lr=0.0005  # Lower learning rate
    )
    
    print(f"Best model found at epoch {best_epoch} with accuracies: {final_accuracies}")
    
    # Visualize model predictions
    print("Visualizing model predictions...")
    visualize_predictions(trained_model, test_loader, device)
    
    # Visualize correct/incorrect distribution
    print("Visualizing correct/incorrect distribution...")
    visualize_correct_incorrect_distribution(trained_model, test_loader, device)
    
    # Visualize confusion matrices
    print("Visualizing confusion matrices...")
    visualize_confusion_matrices(trained_model, test_loader, device)
    
    # Visualize per-fret accuracy
    print("Visualizing per-fret accuracy...")
    visualize_per_fret_accuracy(trained_model, test_loader, device)
    
    # Testing
    print("Testing the trained model...")
    test_model(trained_model, test_loader, device)

if __name__ == "__main__":
    main()
