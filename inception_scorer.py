import pickle

import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import argparse



ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)



def inception_score(images, batch_size=128, resize=True, splits=10):
    """
    Compute Inception Score (IS) for a set of images.

    Parameters:
        images: Tensor of shape (N, C, H, W) in range [-1,1]
        batch_size: batch size for processing images through Inception v3
        resize: whether to resize images to 299x299
        splits: number of splits for IS calculation

    Returns:
        mean IS, std IS
    """
    N = images.size(0)
    device = images.device
    # Scale from [-1,1] to [0,1]
    if images.min() < 0:
        images = (images + 1.0) / 2.0

    # Apply ImageNet normalization
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images = (images - imagenet_mean) / imagenet_std


    # Load pre-trained Inception v3
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # --- Get predictions in batches ---
    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="Calculating inception score"):
            batch = images[i : i + batch_size]
            # Resize batch if needed
            if resize:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

            # Forward pass
            pred = inception_model(batch)
            pred = F.softmax(pred, dim=1)
            all_preds.append(pred.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    
    # Compute IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits):(k+1) * (N // splits), :]    # last images might be discarded since we want equal size split
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py + 1e-8)))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


inception_scores={}
def scorer(file_path):
    all_fake_images = torch.load(file_path, map_location=device)
    # Compute Inception Score
    mean_IS, std_dev = inception_score(all_fake_images, batch_size=128, resize=True, splits=10)
    return mean_IS,std_dev


parser = argparse.ArgumentParser(description="Calculate Inception score")

# 2. Define arguments
parser.add_argument('--img_dir', type=str, help='images directory',required=True)
parser.add_argument('--start_epoch', type=int, help='start epoch',required=True)
parser.add_argument('--end_epoch', type=int, help='end epoch',required=True)
parser.add_argument('--wait', type=str, choices=['y', 'n'], default='n', help='wait for the end epoch file to appear')

# 3. Parse the arguments
args = parser.parse_args()

path=args.img_dir
start=args.start_epoch
end=args.end_epoch

print(f"start epoch: {start}")
print(f"end epoch: {end}")
print(f"path: {path}")

lastFileExists=False
import time
import os

if args.wait=="y":
    while not lastFileExists:
        file_path=f"{path}/all_fake_images_epoch_{end}.pt"
        print(file_path)
        if os.path.exists(file_path):
            lastFileExists=True
        else:
            print(f"File not found for epoch: {end}")
            # Sleep for 1 minute (60 seconds)
            time.sleep(60)
            continue
        
for i in range(start,end+1,5):
    print(f"Epoch: {i}")
    try:
        mean_IS,std_dev=scorer(f"{path}/all_fake_images_epoch_{i}.pt")
    except FileNotFoundError:
        print(f"File not found for epoch: {i}")
        continue
    inception_scores[i] = f"{mean_IS:.4f} +- {std_dev:.4f}"
    print(inception_scores[i])

print(inception_scores)

