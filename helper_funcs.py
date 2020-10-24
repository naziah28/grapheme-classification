from __future__ import print_function, division

import cv2
from tqdm import tqdm_notebook as tqdm
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import pandas as pd 
import glob
import re

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class BengaliDatasetMultiClass(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.label_df = pd.read_csv(csv_file)
        self.label_df = self.label_df[['image_id','grapheme_root',
                             'vowel_diacritic','consonant_diacritic',
                             'label','grapheme','textlabel']]
        
        self.root_dir = root_dir 
        self.transform = transform 
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,
                                self.label_df.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('L')

        label = tuple(self.label_df.iloc[idx, 1:4])
        label = torch.tensor(label)
        textlabel = self.label_df.iloc[idx, -1]  
        
        if self.transform:
            image = self.transform(image)

        return image, label, textlabel
    
    
class BengaliDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.label_df = pd.read_csv(csv_file)
        self.label_df = self.label_df[['image_id','grapheme_root',
                             'vowel_diacritic','consonant_diacritic',
                             'label','grapheme','textlabel']]

        self.root_dir = root_dir 
        self.transform = transform 
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,
                                self.label_df.iloc[idx, 0] + '.png')
        
        image = Image.open(img_name).convert('L')

        label = self.label_df.iloc[idx, 4]             
        textlabel = self.label_df.iloc[idx, -1]  
        if self.transform:
            image = self.transform(image)

        return image, label, textlabel

def plot_gallery(images, titles, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""    
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    titles = titles.numpy()
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        img = np.squeeze(images[i].numpy())
        plt.imshow(img, cmap='gray')
        plt.title(u+titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
def plot_gallery2(images, titles, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""    
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    titles = titles
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        img = np.squeeze(images[i])
        plt.imshow(img, cmap='gray')
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
