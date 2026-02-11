"""
Training script for Deepfake Detection Model
Includes data loading, training loop, validation, and checkpoint management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
from datetime import datetime
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection
    Expects directory structure:
    data/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        
        # Load real samples
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*.jpg'):
                self.samples.append((img_path, 0))  # Label 0 = Real
            for img_path in real_dir.glob('*.png'):
                self.samples.append((img_path, 0))
        
        # Load fake samples
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.jpg'):
                self.samples.append((img_path, 1))  # Label 1 = Fake
            for img_path in fake_dir.glob('*.png'):
                self.samples.append((img_path, 1))
        
        logger.info(f"Loaded {len(self.samples)} samples from {split} set")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float32)


def get_transforms(img_size=256):
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


class ModelTrainer:
    """Training pipeline"""
    
    def __init__(self, model, device='cuda', lr=1e-4, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        
        # Loss with label smoothing
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total
        
        # Compute ROC-AUC
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(all_labels, all_probs)
        except:
            roc_auc = 0.0
        
        return avg_loss, avg_acc, roc_auc
    
    def fit(self, train_loader, val_loader, epochs=50, checkpoint_dir='./checkpoints'):
        """Full training loop"""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_auc = self.validate(val_loader)
            
            self.scheduler.step()
            
            # Record metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}"
            )
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
                self._save_checkpoint(checkpoint_path, epoch)
                logger.info(f"Saved best model to {checkpoint_path}")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = Path(checkpoint_dir) / f'epoch_{epoch}.pth'
                self._save_checkpoint(checkpoint_path, epoch)
        
        # Save final model
        final_path = Path(checkpoint_dir) / 'final_model.pth'
        self._save_checkpoint(final_path, epochs)
        
        logger.info("Training completed!")
        
        return self.history
    
    def _save_checkpoint(self, path, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, path)


def main(
    data_dir='./data',
    output_dir='./experiments',
    epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    img_size=256
):
    """Main training function"""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    output_dir = Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data loading
    logger.info("Loading datasets...")
    train_transform, val_transform = get_transforms(img_size)
    
    train_dataset = DeepfakeDataset(data_dir, split='train', transform=train_transform)
    val_dataset = DeepfakeDataset(data_dir, split='val', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    logger.info("Initializing model...")
    from detector_core import DeepfakeDetectionModel
    model = DeepfakeDetectionModel(pretrained=True)
    
    # Training
    trainer = ModelTrainer(
        model=model,
        device=device,
        lr=learning_rate
    )
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=str(output_dir / 'checkpoints')
    )
    
    # Save training history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved to {history_path}")
    
    # Plot results
    plot_training_curves(history, output_dir)


def plot_training_curves(history, output_dir):
    """Plot training curves"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curve
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=100)
        logger.info(f"Training curves saved")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping curve plotting")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train deepfake detector')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=args.img_size
    )
