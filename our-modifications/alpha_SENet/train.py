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
    EarlyStopping,
)
from config import opt

# Load models
import timm.models.resnet as models

# import timm.models.densenet as models
# import timm.models.efficientnet as models
# import timm.models.regnet as models
# import timm.models.convnext as models
# import timm.models.mobilenetv3 as models


def main(**kwargs):
    opt._parse(kwargs)

    for idx, model_idx in enumerate(opt.model):
        # showing running model
        print(f"RUNNING: {model_idx}")
        print(f"USE_GPUS: {opt.gpu_mode}")
        print(f"PRETRAINED: {opt.pretrain}")
        print(f"OPTIMIZER: {opt.optimizer}")
        print(
            f"LOADING: {opt.load_model_path[idx] if opt.continue_training else opt.continue_training}"
        )
        print(f"VAL_RATE: {opt.val_rate}")
        print(f"EARLYSTOP: {opt.use_earlystop}")

        # prepare device
        device = t.device(
            opt.device if (t.cuda.is_available() and opt.gpu_mode >= 1) else "cpu"
        )
        print(f"On Server: {device}")
        t.backends.cudnn.benchmark = True

        # loadng datasets
        data_transform = {
            "train": T.Compose(
                [
                    # scale: lower&upper ratio bound of original image, [2048/2448]=0.83
                    # ratio: lower&upper aspect ratio of crop area,
                    T.RandomResizedCrop(
                        size=opt.resolution, scale=(0.5, 0.8), ratio=(1, 1)
                    ),
                    # T.RandomResizedCrop(opt.resolution),
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
        ) = read_split_data(opt.train_data_root, val_rate=opt.val_rate)

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
            num_classes=opt.num_classes, pretrained=opt.pretrain
        )
        # print(model)
        model.to(device=device)

        # load model weights
        if opt.continue_training:
            model_weight_path = opt.load_model_path[idx]
            assert os.path.exists(model_weight_path), "cannot find {} file".format(
                model_weight_path
            )

            model.load_state_dict(
                t.load(model_weight_path, map_location=device), strict=True
            )
            model.to(device)

        # loss func, optimizer and lr scheduler
        loss_function = t.nn.CrossEntropyLoss()
        params_groups = get_params_groups(model, weight_decay=opt.weight_decay)
        if opt.optimizer == "SGD":
            optimizer = getattr(optim, opt.optimizer)(
                params_groups,
                lr=opt.learning_rate,
                momentum=0.9,
                weight_decay=opt.weight_decay,
                nesterov=True,
            )
        elif opt.optimizer == "AdamW":
            optimizer = getattr(optim, opt.optimizer)(
                params_groups,
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay,
            )
        else:
            raise EnvironmentError("optimizer can either be SGD or AdamW.")

        # use learning rate scheduler
        if opt.use_lr_scheduler:
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

        dst_str = f"{opt.time_stamp}-{model_idx}-LR_{opt.use_lr_scheduler}_{opt.learning_rate}-BS_{opt.batch_size}-WD_{opt.weight_decay}"
        writer = SummaryWriter(f"runs/{dst_str}")  # init tensorboard

        if opt.use_earlystop:
            earlystop = EarlyStopping(patience=opt.earlystop_patience)  # init earlystop

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
                if opt.use_lr_scheduler:
                    lr_scheduler.step()
                accu_loss += loss.detach()
                train_loader_bar.desc = (
                    "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.7f}".format(
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
                    #                    if train_iteration % 1000 == 0:
                    #                        writer.add_histogram(
                    #                            "fc", model.fc.weight, global_step=train_iteration
                    #                        )
                    #                        writer.add_histogram(
                    #                            "conv5_3-SE_fc2",
                    #                            model.layer4[-1].alpha_attn_mode.fc2.weight,
                    #                            global_step=train_iteration,
                    #                        )
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

            val_epoch_acc = val_accu_num.item() / val_sample_num
            writer.add_scalar(
                accu_tags[3],
                val_epoch_acc,
                epoch + 1,
            )

            # create checkpoint dst
            checkpoint_dst = f"./checkpoints/{dst_str}"
            if os.path.exists(checkpoint_dst) is False:
                os.makedirs(checkpoint_dst)

            # save best model
            if best_acc < val_epoch_acc:
                t.save(model.state_dict(), f"{checkpoint_dst}/best_model.pth")
                best_acc = val_epoch_acc
                with open(f"{checkpoint_dst}/best_model_epoch.txt", "a+") as af:
                    af.write(f"{epoch+1}: {best_acc}\n")

            # save each model after 11-th epoch
            if epoch >= 5:
                t.save(model.state_dict(), f"{checkpoint_dst}/last_model-{epoch+1}.pth")

            # earlystop
            if opt.use_earlystop:
                earlystop(
                    val_loss=val_accu_loss.item() / (val_step + 1),
                    val_acc=val_epoch_acc,
                    model=model,
                    current_epoch=epoch + 1,
                    hyperparameter=dst_str,
                )
                if earlystop.early_stop:
                    print(f"early stop at {epoch-6}...")
                    break

        writer.close()


if __name__ == "__main__":
    main()
