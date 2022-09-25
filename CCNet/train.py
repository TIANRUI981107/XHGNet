from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time

# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import (
    ResidualAttentionModel_92_32input_update as ResidualAttentionModel,
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

if os.path.exists("./save_weights") is False:
    os.makedirs("./save_weights")

tb_writer = SummaryWriter()

model_file = "model_92_sgd.pth"


# for test
def test(model, test_loader, btrain=False, model_file="model_92.pth"):
    # Test
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    for images, labels in test_loader:
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        #
        c = (predicted == labels.data).squeeze()
        for i in range(20):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print(
        "Accuracy of the model on the test images: %d %%"
        % (100 * float(correct) / total)
    )
    print("Accuracy of the model on the test images:", float(correct) / total)
    for i in range(10):
        print(
            "Accuracy of %5s : %2d %%"
            % (classes[i], 100 * class_correct[i] / class_total[i])
        )
    return correct / total


# Image Preprocessing
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4),  # left, top, right, bottom
        # transforms.Scale(224),
        transforms.ToTensor(),
    ]
)
test_transform = transforms.Compose([transforms.ToTensor()])
# when image is rgb, totensor do the division 255
# CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(
    root="./data/", train=True, transform=transform, download=True
)

test_dataset = datasets.CIFAR10(root="./data/", train=False, transform=test_transform)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=256, shuffle=True, num_workers=8  # 64
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=20, shuffle=False
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
model = ResidualAttentionModel().to(device)
print(model)

lr = 0.1  # 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001
)
is_train = True
is_pretrain = False
acc_best = 0
total_epoch = 50
if is_train is True:
    if is_pretrain == True:
        model.load_state_dict((torch.load(model_file)))
    # Training
    for epoch in range(total_epoch):
        train_loss = []
        model.train()
        tims = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.to(device))
            # print(images.data)
            labels = Variable(labels.to(device))

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("hello")
            if (i + 1) % 100 == 0:
                print(
                    "Epoch [%d/%d], Iter [%d/%d] Loss: %.4f"
                    % (
                        epoch + 1,
                        total_epoch,
                        i + 1,
                        len(train_loader),
                        loss.detach().item(),
                    )
                )
            train_loss.append(loss.detach().item())
        train_loss = np.average(train_loss)
        print("the epoch takes time:", time.time() - tims)
        print("evaluate test set:")
        val_acc = test(model, test_loader, btrain=True)

        tags = ["train_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch + 1)
        tb_writer.add_scalar(tags[1], val_acc, epoch + 1)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch + 1)

        if val_acc > acc_best:
            acc_best = val_acc
            print("current best val_acc,", acc_best)
            torch.save(model.state_dict(), f"./save_weights/{model_file}")
        # Decaying Learning Rate
        if (
            (epoch + 1) / float(total_epoch) == 0.3
            or (epoch + 1) / float(total_epoch) == 0.6
            or (epoch + 1) / float(total_epoch) == 0.9
        ):
            lr /= 10
            print("reset learning rate to:", lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                print(param_group["lr"])
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    # Save the Model
    torch.save(model.state_dict(), "./save_weights/last_model_92_sgd.pth")

else:
    test(model, test_loader, btrain=False)
