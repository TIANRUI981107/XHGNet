import sys
from tqdm import tqdm
import os

import torch as t
import torch.optim as optim
import torchvision
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

from data_config import MyDataSet
from utils import (
    read_split_data,
    create_lr_scheduler,
    get_params_groups,
)
from config import opt

# Load models
import timm.models.resnet as models


def main(**kwargs):
    opt._parse(kwargs)

    for idx in range(len(opt.model)):
        # showing running model
        model_idx = opt.model[idx]
        print(f"\nRUNNING: {model_idx}")

        # prepare device
        device = t.device(
            opt.device if (t.cuda.is_available() and opt.use_gpu) else "cpu"
        )
        print(f"On Server: {device}")
        t.backends.cudnn.benchmark = True

        # loadng datasets
        data_transform = {
            "train": T.Compose(
                [
                    T.RandomResizedCrop(opt.resolution),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    T.Normalize(opt.data_mean, opt.data_std),
                ]
            ),
            "val": T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(opt.resolution),
                    T.ToTensor(),
                    T.Normalize(opt.data_mean, opt.data_std),
                ]
            ),
        }

        (
            train_images_path,
            train_images_label,
            val_images_path,
            val_images_label,
        ) = read_split_data(opt.train_data_root, val_rate=0.4)

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

        batch_size = opt.batch_size
        nw = min(
            [os.cpu_count(), batch_size if batch_size > 1 else 0, opt.num_workers]
        )  # number of workers
        print("Using {} dataloader workers every process".format(nw))

        train_loader = t.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=nw,
            collate_fn=train_dataset.collate_fn,
        )

        val_loader = t.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=nw,
            collate_fn=val_dataset.collate_fn,
        )

        # create model
        model = getattr(models, model_idx)(
            num_classes=opt.num_classes, pretrained=False
        )
        # print(model)
        model.to(device=device)

        # loss func, optimizer and lr scheduler
        loss_function = t.nn.CrossEntropyLoss()
        params_groups = get_params_groups(model, weight_decay=opt.weight_decay)
        optimizer = optim.SGD(
            params_groups,
            lr=opt.learning_rate,
            momentum=0.9,
            weight_decay=opt.weight_decay,
            nesterov=True,
        )
        lr_scheduler = create_lr_scheduler(
            optimizer,
            len(train_loader),
            opt.max_epoch,
            warmup=True,
            warmup_epochs=opt.warmup_epochs,
        )

        # training and evaluation
        best_acc = 0.0
        train_iteration = 0
        val_iteration = 0
        writer = SummaryWriter(f"runs/{model_idx}-{opt.time_stamp}")  # init tensorboard
        for epoch in range(opt.max_epoch):
            # train
            model.train()
            accu_loss = t.zeros(1).to(device)  # 累计损失
            accu_num = t.zeros(1).to(device)  # 累计预测正确的样本数
            optimizer.zero_grad()
            sample_num = 0
            train_loader_bar = tqdm(train_loader, file=sys.stdout)
            for step, (images, labels, _, _) in enumerate(train_loader_bar):
                sample_num += images.shape[0]
                pred = model(images.to(device))
                pred_classes = t.max(pred, dim=1)[1]
                accu_num += t.eq(pred_classes, labels.to(device)).sum()
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
                if not t.isfinite(loss):
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
                if opt.debug_mode:
                    if train_iteration == 0:
                        writer.add_graph(model, images.to(device))
                    if (epoch % 10 == 0) and (train_iteration % 100 == 0):
                        img_grid = torchvision.utils.make_grid(images)
                        writer.add_image(
                            "Small-XHGNet/train", img_grid, global_step=train_iteration
                        )
                    if train_iteration % 1000 == 0:
                        writer.add_histogram(
                            "fc", model.fc.weight, global_step=train_iteration
                        )
                        writer.add_histogram(
                            "conv5_3-SE_fc2",
                            model.layer4[-1].alpha_attn_mode.fc2.weight,
                            global_step=train_iteration,
                        )
                    writer.add_scalar(
                        tags[0], loss.detach(), global_step=train_iteration
                    )
                    writer.add_scalar(
                        tags[2],
                        accu_num.item() / sample_num,
                        global_step=train_iteration,
                    )
                    writer.add_scalar(
                        tags[4], optimizer.param_groups[0]["lr"], train_iteration
                    )
                    train_iteration += 1

            # validate
            model.eval()
            with t.no_grad():
                val_accu_num = t.zeros(1).to(device)  # 累计预测正确的样本数
                val_accu_loss = t.zeros(1).to(device)  # 累计损失
                val_sample_num = 0
                val_loader_bar = tqdm(val_loader, file=sys.stdout)
                for val_step, (val_images, val_labels, _, _) in enumerate(
                    val_loader_bar
                ):
                    val_sample_num += val_images.shape[0]
                    val_pred = model(val_images.to(device))
                    val_pred_classes = t.max(val_pred, dim=1)[1]
                    val_accu_num += t.eq(val_pred_classes, val_labels.to(device)).sum()
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
                    if opt.debug_mode:
                        if (epoch % 10 == 0) and (val_iteration % 100 == 0):
                            img_grid_ = torchvision.utils.make_grid(val_images)
                            writer.add_image(
                                "Small-XHGNet/test",
                                img_grid_,
                                global_step=val_iteration,
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

            if os.path.exists(f"./checkpoints/{model_idx}-{opt.time_stamp}") is False:
                os.makedirs(f"./checkpoints/{model_idx}-{opt.time_stamp}")
            if best_acc < val_acc:
                t.save(
                    model.state_dict(),
                    f"./checkpoints/{model_idx}-{opt.time_stamp}/best_model.pth",
                )
                best_acc = val_acc
            if epoch >= (opt.max_epoch - 5):
                t.save(
                    model.state_dict(),
                    f"./checkpoints/{model_idx}-{opt.time_stamp}/last_model-{epoch}.pth",
                )
        writer.flush()


if __name__ == "__main__":

    main()
