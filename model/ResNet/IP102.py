import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import numpy as np

plt.ion()   # interactive mode


base_folder = '../../../data/ip102_v1.1/'

class IP102(Dataset):
    def __init__(self, base_folder=base_folder, transform=None, train=True):
        self.base_folder = base_folder
        self.transform = transform
        self.train = train
        train_images = pd.read_csv(os.path.join(base_folder, 'train.txt'), sep=' ',
                        names=['filepath', 'target'])
        val_images = pd.read_csv(os.path.join(base_folder, 'val.txt'), sep=' ',
                        names=['filepath', 'target'])
        if self.train:
            self.data = train_images
        else:
            self.data = val_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(base_folder, 'images', sample.filepath)
        img = default_loader(path)
        label = sample.target
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def visualize_data():
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        # channel first for image show
        inp = inp.numpy().transpose((1, 2, 0))
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 读取数据
    train_data = IP102(transform=transform, train=True)
    test_data = IP102(transform=transform, train=False)

    trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=8, shuffle=True, num_workers=0)

    # Get a batch of training data
    inputs, classes = next(iter(trainloader))

    # print(inputs[0].size())

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # print(out.size())
    imshow(out)
    # 显示前需要关闭交互模式
    plt.ioff()
    plt.show()

if __name__ == '__main__':

    ip102 = IP102(base_folder)
    img, label = ip102[100]

    print('label : {}'.format(label))
    print('length of ip102 dataset : {}'.format(len(ip102)))

    visualize_data()

  