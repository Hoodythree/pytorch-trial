from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms


import torch
import os
import pandas as pd

base_folder = '../../../data/ip102_v1.1/'

class IP102(Dataset):
    def __init__(self, base_folder=base_folder, transform=None):
        self.base_folder = base_folder
        self.transform = transform

        image_names = os.listdir(os.path.join(base_folder, 'images'))
        df = pd.DataFrame({'filepath':image_names})
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(base_folder, 'images/', sample.filepath)
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
]
)
dataset = IP102(base_folder, transform)
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=0,
    shuffle=False
)


mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print('mean : {}, variance : {}'.format(mean, std))