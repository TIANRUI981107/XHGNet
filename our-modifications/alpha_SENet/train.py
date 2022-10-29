import time
import sys
from tqdm import tqdm
import os
from torchinfo import summary

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from data_config import MyDataSet
from utils import (
    read_split_data,
    create_lr_scheduler,
    get_params_groups,
    plot_data_loader_image,
)
from data_config.constants import SMALL_XHGNET_DEFAULT_MEAN, SMALL_XHGNET_DEFAULT_STD
from config import DefaultConfig

# Load models
# from timm.models.resnet import resnet50 as create_model
# from timm.models.resnet import alpha_resnet50 as create_model
# from timm.models.resnet import resnet50d as create_model
# from timm.models.resnet import resnet50t as create_model
# from timm.models.resnet import resnet50_gn as create_model
# from timm.models.resnet import resnext50_32x4d as create_model
# from timm.models.resnet import alpha_resnext50_32x4d as create_model
# from timm.models.resnet import seresnet50 as create_model
from timm.models.resnet import seresnext50_32x4d as create_model


def main(args):

    device = torch.device(args.device) if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"using {device}")
    writer = SummaryWriter(f"runs/{args.model}-{args.time_stamp}")

    (
        train_images_path,
        train_images_label,
        val_images_path,
        val_images_label,
    ) = read_split_data(args.train_data_root)

    data_transform = {
        "train": transforms.Compose(
            [
                # transforms.Resize(256),
                # transforms.CenterCrop(args.resolution),
                transforms.RandomResizedCrop(args.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    SMALL_XHGNET_DEFAULT_MEAN, SMALL_XHGNET_DEFAULT_STD
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    SMALL_XHGNET_DEFAULT_MEAN, SMALL_XHGNET_DEFAULT_STD
                ),
            ]
        ),
    }

    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"],
    )
    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"],
    )

    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, args.num_workers]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    # plot_data_loader_image(train_loader)

    model = create_model(num_classes=args.num_classes, pretrained=False)

    # print(model)
    # summary(model=model, input_size=(2, 3, 224, 224))
    model.to(device=device)

    loss_function = torch.nn.CrossEntropyLoss()
    params_groups = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = optim.SGD(
        params_groups,
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    lr_scheduler = create_lr_scheduler(
        optimizer,
        len(train_loader),
        args.max_epoch,
        warmup=True,
        warmup_epochs=args.warmup_epochs,
    )

    best_acc = 0.0
    train_iteration = 0
    val_iteration = 0
    for epoch in range(args.max_epoch):

        # train
        model.train()
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        optimizer.zero_grad()

        sample_num = 0
        train_loader_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_loader_bar):
            images, labels = data
            sample_num += images.shape[0]

            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            accu_loss += loss.detach()

            train_loader_bar.desc = (
                "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
                    epoch,
                    accu_loss.item() / (step + 1),
                    accu_num.item() / sample_num,
                    optimizer.param_groups[0]["lr"],
                )
            )

            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)

            # tensorboard debugging
            tags = [
                "Loss/train",
                "Loss/test",
                "Accuracy/train",
                "Accuracy/test",
                "Learning_rate",
            ]
            if args.debug_mode:
                if train_iteration == 0:
                    writer.add_graph(model, images.to(device))
                if (epoch % 10 == 0) and (train_iteration % 100 == 0):
                    img_grid = torchvision.utils.make_grid(images)
                    writer.add_image(
                        "Small-XHGNet/train", img_grid, global_step=train_iteration
                    )
                if train_iteration % 500 == 0:
                    writer.add_histogram(
                        "fc", model.fc.weight, global_step=train_iteration
                    )
                writer.add_scalar(tags[0], loss.detach(), global_step=train_iteration)
                writer.add_scalar(
                    tags[2], accu_num.item() / sample_num, global_step=train_iteration
                )
                writer.add_scalar(
                    tags[4], optimizer.param_groups[0]["lr"], train_iteration
                )
                train_iteration += 1

        # validate
        model.eval()
        with torch.no_grad():
            val_accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
            val_accu_loss = torch.zeros(1).to(device)  # 累计损失

            val_sample_num = 0
            val_loader_bar = tqdm(val_loader, file=sys.stdout)
            for val_step, val_data in enumerate(val_loader_bar):
                val_images, val_labels = val_data
                val_sample_num += val_images.shape[0]

                val_pred = model(val_images.to(device))
                val_pred_classes = torch.max(val_pred, dim=1)[1]
                val_accu_num += torch.eq(val_pred_classes, val_labels.to(device)).sum()

                val_loss = loss_function(val_pred, val_labels.to(device))
                val_accu_loss += val_loss

                val_loader_bar.desc = (
                    "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                        epoch,
                        val_accu_loss.item() / (val_step + 1),
                        val_accu_num.item() / val_sample_num,
                    )
                )

                val_acc = val_accu_num.item() / val_sample_num

                # tensorboard debug
                tags = [
                    "Loss/train",
                    "Loss/test",
                    "Accuracy/train",
                    "Accuracy/test",
                    "Learning_rate",
                ]
                if args.debug_mode:
                    if (epoch % 10 == 0) and (val_iteration % 100 == 0):
                        img_grid_ = torchvision.utils.make_grid(val_images)
                        writer.add_image(
                            "Small-XHGNet/test", img_grid_, global_step=val_iteration
                        )
                    writer.add_scalar(tags[1], val_loss, val_iteration)
                    writer.add_scalar(tags[3], val_acc, val_iteration)
                    val_iteration += 1

        # tensorboard epoch
        accu_tags = [
            "Accum_Loss/train",
            "Accum_Loss/test",
            "Accum_Accuracy/train",
            "Accum_Accuracy/test",
        ]
        writer.add_scalar(accu_tags[0], accu_loss.item() / (step + 1), epoch + 1)
        writer.add_scalar(accu_tags[2], accu_num.item() / sample_num, epoch + 1)
        writer.add_scalar(
            accu_tags[1],
            val_accu_loss.item() / (val_step + 1),
            epoch + 1,
        )
        writer.add_scalar(
            accu_tags[3],
            val_accu_num.item() / val_sample_num,
            epoch + 1,
        )
        writer.close()

        if os.path.exists(f"./checkpoints/{args.model}-{args.time_stamp}") is False:
            os.makedirs(f"./checkpoints/{args.model}-{args.time_stamp}")
        if best_acc < val_acc:
            torch.save(
                model.state_dict(),
                f"./checkpoints/{args.model}-{args.time_stamp}/best_model.pth",
            )
            best_acc = val_acc
        if epoch > (args.max_epoch - 5):
            torch.save(
                model.state_dict(),
                f"./checkpoints/{args.model}-{args.time_stamp}/last_model-{epoch}.pth",
            )
    writer.flush()


if __name__ == "__main__":

    opt = DefaultConfig()

    main(opt)
