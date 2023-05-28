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
        cam_extractor = getattr(methods, cam_method)(model)

        model.eval()

        # with methods.__dict__[cam_method](model) as cam_extractor:
        # with getattr(methods, cam_method)(model) as cam_extractor:
        for test_images, test_labels, test_img_path, test_filename in tqdm(test_loader):
            # t1 = time_synchronized()
            out = model(test_images.to(device))

            argmax_act = out.squeeze(0).argmax().item()  # argmax
            cls_act = test_labels.item()  # true label
            softmax_max = (
                softmax(out, dim=1).squeeze(0)[cls_act].item()
            )  # true label's softmax

            # t2 = time_synchronized()
            activation_map = cam_extractor(cls_act, out)
            zero_act_map = torch.zeros(
                activation_map[0].squeeze(0).size()
            )  # zero mask with same size of act map

            # Resize the CAM and overlay it
            # t3 = time_synchronized()
            img_tensor = read_image(test_img_path[0])
            crop_from_center = transforms.CenterCrop(2048 / 256 * opt.resolution)
            result = overlay_mask(
                to_pil_image(crop_from_center(img_tensor)),
                to_pil_image(activation_map[0].squeeze(0), mode="F"),
                alpha=0.5,
            )
            crop_mask = overlay_mask(
                to_pil_image(crop_from_center(img_tensor)),
                to_pil_image(zero_act_map),
                alpha=0.999,
            )

            # heatmap dir
            dst_mis = os.path.abspath(f"./heatmap/{cam_method}-{opt.time_stamp}/mis")
            if os.path.exists(dst_mis) is False:
                os.makedirs(dst_mis)
            dst_all = os.path.abspath(f"./heatmap/{cam_method}-{opt.time_stamp}/all")
            if os.path.exists(dst_all) is False:
                os.makedirs(dst_all)
            dst_crop = os.path.abspath(f"./heatmap/{cam_method}-{opt.time_stamp}/crop")
            if os.path.exists(dst_crop) is False:
                os.makedirs(dst_crop)
            filename = f"{test_filename[0].split('.')[0]}-{model_i}-softmax_{softmax_max:.5f}-T_{class_indict[str(cls_act)]}-P_{class_indict[str(argmax_act)]}.png"

            # t4 = time_synchronized()
            # save missclassified images
            if argmax_act != cls_act:
                plt.imshow(result)
                plt.axis("off")
                plt.tight_layout()
                # plt.show()
                plt.savefig(
                    os.path.join(dst_mis, filename),
                    dpi=600,
                    bbox_inches="tight",
                )
                plt.clf()
            # save all images
            plt.imshow(result)
            plt.axis("off")
            plt.tight_layout()
            # plt.show()
            plt.savefig(
                os.path.join(dst_all, filename),
                dpi=600,
                bbox_inches="tight",
            )
            plt.clf()
            # save crop ori images
            if model_i == "resnet50":
                plt.imshow(crop_mask)
                plt.axis("off")
                plt.tight_layout()
                # plt.show()
                plt.savefig(
                    os.path.join(dst_crop, filename),
                    dpi=600,
                    bbox_inches="tight",
                )
                plt.clf()

            # t5 = time_synchronized()

            # print(f"model: {t2-t1}\nact: {t3-t2}\noverlay: {t4-t3}\nsave_act: {t5-t4}")


if __name__ == "__main__":
    inference()
