import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, class_to_idx=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx or {
            cls: idx for idx, cls in enumerate(sorted(dataframe["class"].unique()))
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.loc[idx, "filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[self.df.loc[idx, "class"]]

        if self.transform:
            image = self.transform(image)

        return image, label
