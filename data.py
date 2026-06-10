import torch

from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from typing import NamedTuple
ROOT = "./data"


class CelebVAETransform:
    def __init__(self, img_size):
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )

    def __call__(self, img):
        img = self.transform(img)
        return img


class CelebData(NamedTuple):
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def get_celb_split(batch_size: int, split: str, img_size: int, root: str = ROOT):
    dataset = CelebA(
        root=root,
        split=split,
        transform=CelebVAETransform(img_size),
        download=True,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader




def get_data(batch_size: int, img_size: int, root: str = ROOT):
    train = get_celb_split(batch_size, 'train', img_size, root)
    val = get_celb_split(batch_size, 'valid', img_size, root)
    test = get_celb_split(batch_size, 'test', img_size, root)
    return CelebData(train, val, test)


def main():
    data = get_data(batch_size=32, img_size=64)
    print(data.train_loader, data.val_loader, data.test_loader)

if __name__ == "__main__":
    main()
 