import os
from collections import deque

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import efficent_vit

if __name__ == '__main__':
    model = create_model(
        'effi_vit_base_224',
        pretrained=False,
        num_classes=9
    )

    checkpoint = torch.load('best_model.pth', map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    t = [transforms.Resize(
        (224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
    transform = transforms.Compose(t)

    dataset = datasets.ImageFolder('CRC-VAL-HE-7K', transform=transform)

    data_loader_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_correct = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader_test):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            num_correct = (preds == labels).sum().item()

            total_correct += num_correct

    print(
        f'total image:{len(dataset)}, correct:{total_correct}, accuracy:{total_correct / len(dataset)}')
