import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pickle
from scipy import linalg
from tqdm import tqdm
import sys
import torchvision.transforms as transforms
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pickle
import sys


def get_inception_activations(images, device, batch_size=64):
    """
    Extract 2048-dimensional activations from InceptionV3 (pool3 layer).
    """
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = torch.nn.Identity()  # remove classifier
    inception.eval().to(device)

    # ImageNet normalization constants
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    activations = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            if batch.min() < 0:
                batch = (batch + 1.0) / 2.0
            # Normalize with ImageNet stats
            batch = (batch - imagenet_mean) / imagenet_std
            # Resize to Inception input size
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            # Extract activations
            act = inception(batch).detach().cpu().numpy()
            activations.append(act)

    activations = np.concatenate(activations, axis=0)
    return activations

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the FID.
    """
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("FID Warning: adding eps to diagonal of covariances")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def fid_score_full(fake_pickle_path, real_dataloader, device, batch_size=64):

    all_fake_images = torch.load(fake_pickle_path, map_location=device)
    print("Loading full real dataset...")
    real_images = []
    for imgs, _ in tqdm(real_dataloader, desc="Loading real images"):
        real_images.append(imgs)
    real_images = torch.cat(real_images, dim=0).to(device)

    # Compute Inception features
    print("Extracting Inception activations...")
    act_fake = get_inception_activations(all_fake_images, device, batch_size)
    act_real = get_inception_activations(real_images, device, batch_size)

    # Compute mean and covariance of activations
    mu_fake, sigma_fake = np.mean(act_fake, axis=0), np.cov(act_fake, rowvar=False)
    mu_real, sigma_real = np.mean(act_real, axis=0), np.cov(act_real, rowvar=False)

    # Compute FID
    fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"FID Score (Full Real Dataset): {fid:.3f}")
    return fid


if __name__ == "__main__":


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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1,1]
    ])

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 64

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 32
    ngpu=1

    dataset = dset.CIFAR10(root=path+"/data", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=workers
    )

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    fid_scores={}
    for epoch in range(start,end+1,5):
        fid_scores[epoch]=fid_score_full(f"{path}/all_fake_images_epoch_{epoch}.pt",dataloader,device)
        print(fid_scores[epoch])
    
    for key,value in fid_scores.items():
        print(f"{key}: {value}")

    filename = path+"/fid_scores.txt"
    with open(filename, 'a') as f:
        for key, value in fid_scores.items():
            f.write(f"{key}: {value}\n")
