import os
import json
from unicodedata import name

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd

from model import convnext_base as create_model


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

    def summary(self):
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
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
            info_csv.append([self.labels[i], Precision, Recall, Specificity])
        print(table)

        try:
            # record info with Pandas
            info_df = pd.DataFrame(np.array(info_csv), columns=['Classes', 'Precision', 'Recall', 'Specificity'])  
            new_info_df = pd.concat([info_df, acc_series], axis=1)
            new_info_df.to_csv("./metrices.csv", index=False)
            print("./info.csv saved successfully!")
        except Exception as e:
            print(e)
            exit(-1)

    def plot(self):
        matrix = self.matrix
        matrix_df = pd.DataFrame(matrix, index=self.labels, columns=self.labels)  # record CFmatrix with pandas
        matrix_df.to_csv("./confusion-matrix.csv", index=False)

        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45, fontsize=8)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                # if info > 100, dont print
                if x != y and info != 0:
                    plt.text(x, y, info,
                             verticalalignment='center',
                             horizontalalignment='center',
                             color="white" if info > thresh else "black",
                             fontsize=8)
        plt.tight_layout()
        plt.savefig("./confusion-matrix.png", dpi=600, bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.8, 0.83), ratio=(0.98, 1.02)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "..", "val"))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(data_root), "data path {} does not exist.".format(data_root)

    validate_dataset = datasets.ImageFolder(root=data_root, transform=data_transform)

    batch_size = 256
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    net = create_model(num_classes=59)

    # load pretrain weights
    model_weight_path = "./best_model_convnext_base_22k_224.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    net.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=34, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()

