import os
import random
import json
import numpy as np
import torch

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for deterministic behavior (might slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_logs(log_data, log_file):
    """Append JSON log entry to file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_data) + '\n')
