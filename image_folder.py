import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BaseImageFolder(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))

    def getitem(self, index, w, h):
        path = self.files[index % len(self.files)]
        img = Image.open(path)
        img = img.resize((w*128, h*128))
        # To ensure that the input image is RGB mode
        # For Ex. ADE_train_00008455.jpg is not an RGB image.
        # It will cause axes don't match in numpy.transpose().
        img = img.convert('RGB')
        img = np.array(img)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        patches = np.reshape(img, (3, h, 128, w, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        return img, patches, path

    def __getitem__(self, index):
        pass

    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)


# Image shape is 6x10 128x128 patches
class ImageFolder720p(BaseImageFolder):
    def __getitem__(self, index):
        return self.getitem(index=index, w=10, h=6)


# Image shape is 8x16 128x128 patches
class ImageFolder2K(BaseImageFolder):
    def __getitem__(self, index):
        return self.getitem(index=index, w=16, h=8)


# Image shape is 8x16 128x128 patches
class ImageFolder1024sqr(BaseImageFolder):
    def __getitem__(self, index):
        return self.getitem(index=index, w=16, h=16)


# Automatically resize input image
class ImageFolderAuto(BaseImageFolder):
    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        (w, h) = Image.open(path).size
        return self.getitem(index=index, w=w//128, h=h//128)
