import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from IP102 import IP102


base_folder = '../../../data/ip102_v1.1/'

class IP102(Dataset):
    def __init__(self, base_folder=base_folder, transform=None):
        self.base_folder = base_folder
        self.transform = transform
        self.train = train
        test_images = pd.read_csv(os.path.join(base_folder, 'test.txt'), sep=' ',
                        names=['filepath', 'target'])
        self.data = test_images

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

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # mean : tensor([0.5139, 0.5406, 0.3881]), variance : tensor([0.1889, 0.1877, 0.1880])
        transforms.Normalize([0.5139, 0.5406, 0.3881], [0.1889, 0.1877, 0.1880])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5139, 0.5406, 0.3881], [0.1889, 0.1877, 0.1880])
    ]),
}

test_datasets = IP102(base_folder=base_folder, transform=data_transforms['val'])
test_dataloader = torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=4)
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_accuracy(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)

            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
        model.train()
        return 100 * correct / total



if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 102)
    model = model_ft.to(device)

    model.load_state_dict(torch.load('resnet_linear.pth'))
    acc = test_accuracy(model, test_dataloader)
    print('test acc : ', acc)
    
    

    