import os
import torch

from CV.models.i3d import InceptionI3d


def main():
    cv_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(cv_dir, "checkpoints")

    input_checkpoint = os.path.join(checkpoints_dir, "wlasl100_best_model_75.15pct.pth")
    output_checkpoint = os.path.join(checkpoints_dir, "wlasl100_best_model_75.15pct_FULL.pth")

    if not os.path.exists(input_checkpoint):
        raise FileNotFoundError(f"Input checkpoint not found: {input_checkpoint}")

    print("=" * 60)
    print("RE-SAVING WLASL100 CHECKPOINT")
    print("=" * 60)
    print(f"\n📥 Loading original checkpoint...\n   Path: {input_checkpoint}")

    checkpoint = torch.load(input_checkpoint, map_location="cpu")

    print("\n📦 Original checkpoint keys:")
    for k in checkpoint.keys():
        print(f"   • {k}")

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'model_state_dict' or 'state_dict'")

    num_classes = checkpoint.get("num_classes", 100)

    print("\n🔧 Creating I3D model architecture...")
    model = InceptionI3d(num_classes=num_classes, in_channels=3)

    incompatible = model.load_state_dict(state_dict, strict=False)
    print("📥 Loaded model weights")
    print(f"   • Missing keys: {len(incompatible.missing_keys)}")
    print(f"   • Unexpected keys: {len(incompatible.unexpected_keys)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params:,} parameters, {num_classes} classes")

    new_checkpoint = dict(checkpoint)
    new_checkpoint["model"] = model
    new_checkpoint["num_classes"] = num_classes

    print("\n💾 Saving new checkpoint with full model object...")
    torch.save(new_checkpoint, output_checkpoint)
    print(f"✅ Saved to: {output_checkpoint}")


if __name__ == "__main__":
    main()
