# PyTorch Pretrained CNN

Train image classifiers using PyTorch and pretrained CNN models (ResNet, DenseNet, EfficientNet, etc.) with minimal setup and fully modular code.

---

## Features

- Supports pretrained models via `torchvision`
- Configurable data augmentations (rotation, flipping)
- Confusion matrix + classification report
- Simple YAML-based config system
- Built for image classification tasks

---

## 🗂️ Folder Structure

```
pytorch-pretrained-CNN/
├── src/               # Core code (datasets, training, model creation, etc.)
├── outputs/           # Saved models, logs, confusion matrices
├── config.yaml        # Training configuration
├── train.py           # Main script
├── requirements.txt
└── README.md
```

---

## Downloadable Sample Dataset

> Replace this with your actual data  
Example (2-class folder structure):

```
/train/
  ├── class1/
  │   ├── img1.jpg
  │   ├── img2.jpg
  └── class2/
      ├── img3.jpg
      ├── img4.jpg
```

**📎 Sample Dataset Download (if applicable):**  
[Download Sample Dataset](https://example.com/sample-dataset.zip) ← replace this link

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/pytorch-pretrained-CNN.git
cd pytorch-pretrained-CNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Update config.yaml with your dataset path and preferences

# 4. Train the model
python train.py --config config.yaml
```

---

## Example Config (`config.yaml`)
```yaml
model: densenet201
batch_size: 32
num_classes: 2
image_size: [224, 224]
dataset_path: /path/to/train
horizontal_flip: true
rotation: 30
learning_rate: 0.004
num_epochs: 10
output_dir: outputs/
```

---

## Output Example

- `outputs/model_name/model.pth` — full model
- `outputs/model_name/logs.json` — training logs
- `outputs/model_name/confusion_matrix.png` — confusion matrix

---

## Future Ideas

-  Add support for `timm` models
-  Add learning rate scheduler
-  Add test-time augmentation (TTA)
-  Export to ONNX or TorchScript

