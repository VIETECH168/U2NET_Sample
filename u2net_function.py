import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset
import cv2


def normalize_prediction(image_tensor):
    """Normalize the predicted image tensor to the range [0, 1]."""
    image_num = image_tensor.size(0)
    image_tensor = image_tensor.clone().detach()
    image_min = torch.min(image_tensor.view(image_num, -1), dim=1)[0]
    image_max = torch.max(image_tensor.view(image_num, -1), dim=1)[0]
    image_tensor = (image_tensor - image_min[:, None, None, None]) / (image_max[:, None, None, None] - image_min[:, None, None, None])
    return image_tensor

def save_images(image_tensor, mask_paths, save_path):
    """Save images after normalization, resizing them to match the original mask size."""
    image_num = image_tensor.size(0)
    images = (normalize_prediction(image_tensor) * 255).clone().detach().permute(0, 2, 3, 1).cpu().numpy()

    for i in range(image_num):
        mask_shape = cv2.imread(mask_paths[i]).shape[:2]
        resized_image = cv2.resize(images[i], dsize=(mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_LINEAR)
        save_filename = os.path.join(save_path, os.path.basename(mask_paths[i]))
        cv2.imwrite(save_filename, resized_image)

def calculate_iou_and_save(image_tensor, target_tensor, mask_paths, save_path):
    """Calculate Intersection over Union (IoU) and save results."""
    pred = 1 - torch.round(image_tensor.clone().detach()).long()
    target = 1 - torch.round(target_tensor.clone().detach()).long()

    intersection = torch.sum(pred & target, dim=(1, 2, 3)).float()
    union = torch.sum(pred | target, dim=(1, 2, 3)).float()
    ious = intersection / union

    txt_path = os.path.join(save_path, 'iou.txt')
    new_data = np.c_[np.array([os.path.basename(x) for x in mask_paths]), ious.detach().cpu().numpy()]

    if os.path.isfile(txt_path):
        existing_data = pd.read_table(txt_path, sep=' ', header=None).values
        new_data = np.r_[existing_data, new_data]

    pd.DataFrame(new_data).to_csv(txt_path, sep=' ', index=False, header=False)

def calculate_bce_loss(d_list, target):
    """Calculate the BCE loss across multiple U-Net outputs."""
    criterion = nn.BCELoss()
    losses = [criterion(d, target) for d in d_list]
    total_loss = sum(losses)
    return losses[0], total_loss, *losses[1:]

class SegmentationDataset(Dataset):
    """Custom Dataset class for loading images and masks."""
    def __init__(self, data_path, mask_path, resize=512, data_postfix='.jpg', mask_postfix='.jpg'):
        self.data_paths = self.get_file_paths(data_path, data_postfix)
        self.mask_paths = self.get_file_paths(mask_path, mask_postfix, self.data_paths)
        self.resize = resize

    def __getitem__(self, idx):
        image = io.imread(self.data_paths[idx])[:, :, :3]
        image = transform.resize(image, (self.resize, self.resize), mode='constant') / np.max(image)
        
        mask = io.imread(self.mask_paths[idx])
        if len(mask.shape) == 2:  # Grayscale mask
            mask_resized = transform.resize(mask, (self.resize, self.resize), mode='constant', order=0, preserve_range=True)
        else:  # RGB mask, use the first channel
            mask_resized = transform.resize(mask[:, :, 0], (self.resize, self.resize), mode='constant', order=0, preserve_range=True)

        mask_resized = mask_resized / np.max(mask_resized)
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # Normalize image

        return torch.tensor(image, dtype=torch.float).permute(2, 0, 1), torch.tensor(mask_resized, dtype=torch.float).unsqueeze(0), self.mask_paths[idx]

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def get_file_paths(path, postfix, reference_list=None):
        """Get the file paths for the dataset."""
        root = os.getcwd()
        file_list = []

        if reference_list is None:
            for file in os.listdir(path):
                if os.path.splitext(file)[-1] == postfix:
                    file_list.append(os.path.join(root, path, file))
        else:
            for ref in reference_list:
                file_list.append(os.path.join(root, path, os.path.splitext(os.path.split(ref)[1])[0] + postfix))

        return file_list