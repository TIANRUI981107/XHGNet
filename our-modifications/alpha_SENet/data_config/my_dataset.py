from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        path = self.images_path[idx]
        filename = self.images_path[idx].split("/")[-1]

        img = Image.open(path)
        # check whether RGB images format
        if img.mode != "RGB":
            raise ValueError("image: {} isn't RGB mode.".format(path))
        label = self.images_class[idx]

        if self.transform is not None:
            img = self.transform(img)

        batches = (
            img,
            label,
            path,
            filename,
        )

        return batches

    @staticmethod
    def collate_fn(batch):
        # FIXME: bugs here

        images, labels, img_paths, filenames = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, img_paths, filenames
