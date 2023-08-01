import shutil
import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd

from torchcam.methods import SmoothGradCAMpp
from data_config.my_dataset import MyDataSet
from utils.utils import read_split_data
from config import opt

import timm.models.resnet as models

# import timm.models.densenet as models
# import timm.models.efficientnet as models
# import timm.models.regnet as models
# import timm.models.convnext as models
# import timm.models.mobilenetv3 as models


class ConfusionMatrix(object):
    """
    Draw confusion matrix.

    Note:
        prettytable is needed.
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self, dst):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        acc_series = pd.Series(acc, name="accuracy")  # record acc with Pandas
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        info_csv = []
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.0
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.0
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.0
            table.add_row([self.labels[i], Precision, Recall, Specificity])
            info_csv.append([self.labels[i], Precision, Recall, Specificity])
        print(table)

        try:
            # record info with Pandas
            info_df = pd.DataFrame(
                np.array(info_csv),
                columns=["Classes", "Precision", "Recall", "Specificity"],
            )
            new_info_df = pd.concat([info_df, acc_series], axis=1)
            new_info_df.to_csv(f"{dst}/metrices.csv", index=False)
            print("Evaluation Metrices saved successfully!")
        except Exception as e:
            print(e)
            exit(-1)

    def plot(self, dst):
        matrix = self.matrix
        matrix_df = pd.DataFrame(
            matrix, index=self.labels, columns=self.labels
        )  # record CFmatrix with pandas
        matrix_df.to_csv(f"{dst}/confusion-matrix.csv", index=False)

        print(matrix)
        fig_, ax_ = plt.subplots()
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45, fontsize=8)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Confusion matrix")

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                # if info > 100, dont print
                if x != y and info != 0:
                    plt.text(
                        x,
                        y,
                        info,
                        verticalalignment="center",
                        horizontalalignment="center",
                        color="white" if info > thresh else "black",
                        fontsize=6,
                    )
        plt.tight_layout()
        plt.savefig(f"{dst}/confusion-matrix.png", dpi=800, bbox_inches="tight")
        # plt.show()


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def plot_inference_nms_time(times, dst: str):
    try:
        x = list(range(len(times)))
        fig, ax = plt.subplots()
        plt.plot(x, times, label="Inference Time")
        plt.xlabel("Images/it")
        plt.ylabel("Time/s")
        plt.title("Inference Time of Test Datasets")
        plt.xlim(0, len(times))
        plt.legend(loc="best")
        plt.savefig(f"{dst}/inference_time.png")
        plt.close()
        print("successful save inference_time curve!")
    except Exception as e:
        print(e)


def test(**kwargs):
    opt._parse(kwargs)

    for i in range(len(opt.model)):
        # show running model
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

        # load class_indices
        json_label_path = "./class_indices.json"
        assert os.path.exists(json_label_path), "cannot find {} file".format(
            json_label_path
        )
        json_file = open(json_label_path, "r")
        class_indict = json.load(json_file)

        labels = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=opt.num_classes, labels=labels)

        timer_list = []
        model.eval()
        with torch.no_grad():
            for test_images, test_labels, test_img_path, test_filename in tqdm(
                test_loader
            ):
                img_height, img_width = test_images.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                t_start = time_synchronized()
                outputs = model(test_images.to(device))
                t_end = time_synchronized()
                timer_list.append(t_end - t_start)

                outputs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)

                if pred.item() != test_labels.item():
                    destination = os.path.abspath(
                        f"./hard_example_mining/{opt.time_stamp}-{model_i}-{model_load_path_i.split('/')[-2]}"
                    )
                    if os.path.exists(destination) is False:
                        os.makedirs(destination)

                    shutil.copy(
                        test_img_path[0],
                        f"{destination}/T_{class_indict[str(test_labels.item())]}_P_{class_indict[str(pred.item())]}_{test_filename[0]}",
                    )

                confusion.update(pred.to("cpu").numpy(), test_labels.to("cpu").numpy())
        confusion.plot(dst=destination)
        confusion.summary(dst=destination)
        if len(timer_list) != 0:
            plot_inference_nms_time(timer_list, dst=destination)
            try:
                df = pd.Series(timer_list)
                df.to_csv(f"{destination}/inference_time.csv")
            except Exception as e:
                print(e)
                exit(-1)


if __name__ == "__main__":
    test()
