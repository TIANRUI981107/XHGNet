import time
import json
import os
import torch
from torchvision.io.image import read_image
from torchvision import transforms
from torch.nn.functional import softmax
from torchvision.transforms.functional import to_pil_image
from torchcam import methods
from torchcam.utils import overlay_mask

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_config.my_dataset import MyDataSet
from utils.utils import read_split_data
from config import opt

import timm.models.resnet as models

# import timm.models.densenet as models
# import timm.models.efficientnet as models
# import timm.models.regnet as models
# import timm.models.convnext as models
# import timm.models.mobilenetv3 as models


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def inference(**kwargs):
    opt._parse(kwargs)

    for i in range(len(opt.model)):
        # show running model
        cam_method = opt.cam_method
        model_i = opt.model[i]
        model_load_path_i = opt.load_model_path[i]
        print(f"running model: {model_i}\nloading path: {model_load_path_i}")

        # prepare device
        device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
        print(f"On Server: {device}")

        # load dataset
        data_root = os.path.abspath(opt.test_data_root)
        assert os.path.exists(data_root), "data path {} does not exist.".format(
            data_root
        )

        data_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(opt.resolution),
                transforms.ToTensor(),
                transforms.Normalize(opt.data_mean, opt.data_std),
            ]
        )

        test_images_path, test_images_label = read_split_data(data_root, val_rate=0.0)

        test_dataset = MyDataSet(
            images_path=test_images_path,
            images_class=test_images_label,
            transform=data_transform,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=opt.num_workers,
            collate_fn=test_dataset.collate_fn,
        )
        # load class_indices
        json_label_path = "./class_indices.json"
        assert os.path.exists(json_label_path), "cannot find {} file".format(
            json_label_path
        )
        json_file = open(json_label_path, "r")
        class_indict = json.load(json_file)

        # create_model
        model = getattr(models, model_i)(num_classes=opt.num_classes, pretrained=False)
        # print(model)
        model.to(device=device)

        # load model weights
        model_weight_path = model_load_path_i
        assert os.path.exists(model_weight_path), "cannot find {} file".format(
            model_weight_path
        )

        model.load_state_dict(
            torch.load(model_weight_path, map_location=device), strict=True
        )
        model.to(device)

        model.eval()
        for test_images, test_labels, test_img_path, test_filename in tqdm(test_loader):
            out = model(test_images.to(device))

            softmax_prob = softmax(out, dim=1).squeeze(0).tolist()  # softmax probablity
            softmax_prob.append(test_filename[0])
            softmax_prob.append(test_labels.item())

            # heatmap dir
            filename = f"{model_i}-{opt.time_stamp}.txt"
            with open(filename, "a+") as af:
                af.write(f"{softmax_prob}\n")


if __name__ == "__main__":
    inference()
