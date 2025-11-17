"""
Upload training history from saved checkpoint to Weights & Biases
Retroactively log all training metrics for visualization
"""

import torch
import wandb
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Path to your checkpoint
CHECKPOINT_PATH = r"c:\Users\KH&H\NHA-057\CV\checkpoints\best_model_citizen100_87pct.pth"

# W&B project configuration
PROJECT_NAME = "Sign_Bridge"
RUN_NAME = "citizen100-i3d-87.68pct"
TAGS = ["i3d", "citizen-100", "sign-language", "retroactive-upload"]

# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("="*60)
print("LOADING CHECKPOINT")
print("="*60)

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

print(f"✅ Checkpoint loaded successfully!")
print(f"\n📦 Checkpoint contains:")
for key in checkpoint.keys():
    print(f"   • {key}")

# ============================================================
# EXTRACT TRAINING HISTORY
# ============================================================

history = checkpoint.get('history', {})

if not history:
    print("\n❌ No training history found in checkpoint!")
    print("💡 The checkpoint might not have saved the history.")
    exit(1)

print(f"\n📊 Found training history:")
print(f"   • Epochs trained: {len(history.get('train_loss', []))}")
print(f"   • Best epoch: {checkpoint.get('epoch', 'Unknown')}")

# Get best val accuracy from history if not in checkpoint
best_val_acc = checkpoint.get('best_val_acc')
if best_val_acc is None and 'val_acc' in history and history['val_acc']:
    best_val_acc = max(history['val_acc'])

if best_val_acc is not None:
    print(f"   • Best val accuracy: {best_val_acc:.2f}%")
else:
    print(f"   • Best val accuracy: Unknown")

# ============================================================
# INITIALIZE W&B
# ============================================================

print("\n" + "="*60)
print("INITIALIZING WEIGHTS & BIASES")
print("="*60)
print("\n🔐 You'll be prompted to login if not already authenticated")
print("📝 Get your API key from: https://wandb.ai/authorize\n")

# Initialize W&B run
run = wandb.init(
    project=PROJECT_NAME,
    name=RUN_NAME,
    tags=TAGS,
    config={
        "architecture": "I3D (InceptionI3d)",
        "dataset": "Citizen-100",
        "classes": 100,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-8,
        "scheduler": "ReduceLROnPlateau",
        "input_shape": "(B, 3, 32, 224, 224)",
        "augmentations": 7,
        "best_epoch": checkpoint.get('epoch', 0),
        "best_val_acc": max(history.get('val_acc', [0])) if history.get('val_acc') else 0,
    }
)

print(f"✅ W&B initialized!")
print(f"🌐 Dashboard: {run.get_url()}")

# ============================================================
# UPLOAD TRAINING HISTORY
# ============================================================

print("\n" + "="*60)
print("UPLOADING TRAINING HISTORY")
print("="*60)

num_epochs = len(history.get('train_loss', []))

for epoch in range(num_epochs):
    # Prepare metrics for this epoch
    metrics = {
        'epoch': epoch + 1,
    }
    
    # Add training metrics
    if 'train_loss' in history and epoch < len(history['train_loss']):
        metrics['train/loss'] = history['train_loss'][epoch]
    
    if 'train_acc' in history and epoch < len(history['train_acc']):
        metrics['train/accuracy'] = history['train_acc'][epoch]
    
    # Add validation metrics
    if 'val_loss' in history and epoch < len(history['val_loss']):
        metrics['val/loss'] = history['val_loss'][epoch]
    
    if 'val_acc' in history and epoch < len(history['val_acc']):
        metrics['val/accuracy'] = history['val_acc'][epoch]
    
    # Add learning rate if available
    if 'learning_rates' in history and epoch < len(history['learning_rates']):
        metrics['learning_rate'] = history['learning_rates'][epoch]
    
    # Log to W&B
    wandb.log(metrics, step=epoch)
    
    # Progress indicator
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        print(f"📤 Uploaded epoch {epoch + 1}/{num_epochs}")

# ============================================================
# UPLOAD MODEL SUMMARY
# ============================================================

print("\n" + "="*60)
print("FINALIZING")
print("="*60)

# Log best model info
summary_data = {
    "best_epoch": checkpoint.get('epoch', 0),
    "total_epochs": num_epochs,
    "model_size_mb": os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024),
}

# Add best metrics from history
if 'val_acc' in history and history['val_acc']:
    summary_data["best_val_accuracy"] = max(history['val_acc'])
if 'val_loss' in history and history['val_loss']:
    summary_data["best_val_loss"] = min(history['val_loss'])

wandb.run.summary.update(summary_data)

# Save the checkpoint as an artifact (optional but recommended)
best_val_acc_for_artifact = max(history.get('val_acc', [0])) if history.get('val_acc') else 0

artifact = wandb.Artifact(
    name='citizen100-i3d-model',
    type='model',
    description=f'I3D model trained on Citizen-100 dataset ({best_val_acc_for_artifact:.2f}% val accuracy)',
    metadata={
        'best_epoch': checkpoint.get('epoch', 0),
        'val_accuracy': best_val_acc_for_artifact,
        'classes': 100,
    }
)
artifact.add_file(CHECKPOINT_PATH)
run.log_artifact(artifact)

print(f"✅ All metrics uploaded successfully!")
print(f"📊 Total epochs logged: {num_epochs}")
print(f"🌐 View your training curves at: {run.get_url()}")

# Finish the run
wandb.finish()

print("\n" + "="*60)
print("✅ UPLOAD COMPLETE!")
print("="*60)
print(f"\n🎉 Your training history is now on W&B!")
print(f"📈 You can now visualize:")
print(f"   • Training/Validation loss curves")
print(f"   • Training/Validation accuracy curves")
print(f"   • Learning rate schedule")
print(f"   • Compare with future experiments")
print(f"\n🌐 Dashboard: https://wandb.ai/{wandb.api.default_entity()}/{PROJECT_NAME}")
