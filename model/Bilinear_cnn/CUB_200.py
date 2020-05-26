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


base_folder = '../../data/CUB_200_2011/'

class CUB200(Dataset):
    def __init__(self, base_folder=base_folder, transform=None, train=True):
        self.base_folder = base_folder
        self.transform = transform
        self.train = train
        images = pd.read_csv(os.path.join(base_folder, 'images.txt'), sep=' ',
                        names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(base_folder, 'image_class_labels.txt'),
                                    sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(base_folder, 'train_test_split.txt'),
                                        sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = data[data.is_training_img == 1]
        else:
            self.data = data[data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(base_folder, 'images', sample.filepath)
        img = default_loader(path)
        # Target start from 1 in data, so shift to 0
        label = sample.target - 1
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
    train_data = CUB200(transform=transform, train=True)
    test_data = CUB200(transform=transform, train=False)

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

    cub200 = CUB200(base_folder)
    img, label = cub200[100]

    print('label : {}'.format(label))
    print('length of cub200 dataset : {}'.format(len(cub200)))

    visualize_data()

  