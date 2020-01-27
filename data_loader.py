import pandas as pd
import numpy as np
import warnings

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

def get_data_from_path(entries_path, split):

    data = pd.read_csv(f'{entries_path}/{split}_entries.csv')[['label','xray_paths','text']]
    # Adjusting labels to fit with Snorkel MeTaL labeling convention (0 reserved for abstain)
    data['label'][data['label']==0] = 2
    perc_pos = sum(data['label']==1)/len(data)
    print(f'{len(data)} {split} examples: {100*perc_pos:0.1f}% Abnormal')
        
    return data

class OpenI(Dataset):
    def __init__(self, split, entries_path, transform):
        """
        Args:
            split (string): train/test/split
            entries_path (string): path to folder containing csv of all image paths and text reports
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.transform = transform
        self.entries = get_data_from_path(entries_path, split)
        
    def __getitem__(self, index):
        return np.array(self.entries.iloc[0,:])
    
    def __len__(self):
        return len(self.entries)

class WikiSatNet(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.img_center = 500
        self.embedding_size = 300
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # Second column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column contains the crop sizes
        self.crop_arr = np.asarray(self.data_info.iloc[:, 2])
        # First column is the labels
        self.textual_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        single_crop_size = int(self.crop_arr[index])
        single_textual_embedding_name = self.textual_arr[index]
        # Load image
        img_as_img = Image.open(single_image_name).convert('RGB').crop((self.img_center-single_crop_size/2.,
                    self.img_center-single_crop_size/2., self.img_center+single_crop_size/2., self.img_center+single_crop_size/2.))
        # Load the Textual Embeddings
        textual_embedding_as_tensor = np.load(single_textual_embedding_name).reshape((self.embedding_size))
        img_as_tensor = self.transforms(img_as_img)

        return (img_as_tensor, textual_embedding_as_tensor)

    def __len__(self):
        return self.data_len

class fMoW(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
