import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import argparse

from src.dataset import CustomImageDataset
from src.transforms import get_train_transforms, get_test_transforms
from src.model_builder import create_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import set_seed

# --- Load config ---
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    # --- Config values ---
    dataset_path = config["dataset_path"]
    model_name = config["model"]
    image_type = config.get("image_type", "img")
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    lr = config["learning_rate"]
    num_epochs = config["num_epochs"]
    rotation = config.get("rotation", 0)
    flip = config.get("horizontal_flip", False)

    model_id = f"{image_type}_{model_name}_lr={lr}_bs={batch_size}"
    save_dir = os.path.join(config["output_dir"], model_id)
    os.makedirs(save_dir, exist_ok=True)

    full_model_path = os.path.join(save_dir, "model.pth")
    weights_path = os.path.join(save_dir, "weights.pth")
    logs_path = os.path.join(save_dir, "logs.json")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")

    # --- Prepare dataset ---
    class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    data = []

    for cls in class_names:
        cls_path = os.path.join(dataset_path, cls)
        files = os.listdir(cls_path)
        for f in files:
            data.append((os.path.join(cls, f), cls))

    df = pd.DataFrame(data, columns=["filename", "class"])

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["class"])

    class_to_idx = {cls: i for i, cls in enumerate(sorted(train_df["class"].unique()))}

    train_dataset = CustomImageDataset(train_df, dataset_path, transform=get_train_transforms(
        image_size=tuple(config["image_size"]), horizontal_flip=flip, rotation_range=rotation), class_to_idx=class_to_idx)

    val_dataset = CustomImageDataset(val_df, dataset_path, transform=get_test_transforms(
        image_size=tuple(config["image_size"])), class_to_idx=class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model = train_model(model, train_loader, val_loader, criterion, optimizer, device,
                        num_epochs=num_epochs, full_model_path=full_model_path,
                        weights_path=weights_path, logs_path=logs_path)

    # --- Evaluation ---
    evaluate_model(model, val_loader, class_names=sorted(class_to_idx.keys()), device=device, save_path=cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)
