import os
import cv2
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

r"""
Simple Dataloader for mnist.
"""

class MnistDataset(Dataset):
    def __init__(self, split, im_path, im_ext='png', im_channels=1):
        self.split = split
        self.im_ext = im_ext
        self.im_channels = im_channels
        self.images, self.labels = self.load_images(im_path)
    
    def load_images(self, im_path):
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        assert self.im_channels == 1 or self.im_channels == 3, "Input iamge channels can only be 1 or 3"
        if self.im_channels == 1:
            im = cv2.imread(self.images[index], 0)
        else:
            # Generate a random color digit
            im_1 = cv2.imread(self.images[index], 0)[None, :]*np.clip(random.random(), 0.2, 1.0)
            im_2 = cv2.imread(self.images[index], 0)[None, :]*np.clip(random.random(), 0.2, 1.0)
            im_3 = cv2.imread(self.images[index], 0)[None, :]*np.clip(random.random(), 0.2, 1.0)
            im = np.concatenate([im_1, im_2, im_3], axis=0)
        
        label = self.labels[index]
        # Convert to 0 to 255 into -1 to 1
        im = 2 * (im / 255) - 1
        im_tensor = torch.from_numpy(im)[None, :] if self.im_channels == 1 else torch.from_numpy(im)
        return im_tensor, torch.as_tensor(label)


if __name__ == '__main__':
    mnist = MnistDataset('test', 'data/test/images', im_channels=3)
    mnist_loader = DataLoader(mnist, batch_size=16, shuffle=True, num_workers=0)
    for im, label in mnist_loader:
        print('Image dimension', im.shape)
        print('Label dimension: {}'.format(label.shape))
        break


