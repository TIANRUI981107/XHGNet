import os
from torchinfo import summary

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from data_config import MyDataSet
from utils import (
    read_split_data,
    create_lr_scheduler,
    get_params_groups,
    train_one_epoch,
    evaluate,
    plot_data_loader_image,
)
from data_config.constants import SMALL_XHGNET_DEFAULT_MEAN, SMALL_XHGNET_DEFAULT_STD
from config import DefaultConfig

# Load models
from timm.models.resnet import resnet50 as create_model

# from timm.models.resnet import resnet50d as create_model
# from timm.models.resnet import resnet50t as create_model
# from timm.models.resnet import resnet50_gn as create_model
# from timm.models.resnet import resnetrs50 as create_model
# from timm.models.resnet import seresnet50 as create_model
# from timm.models.resnet import seresnext50_32x4d as create_model


def main(args):

    device = torch.device(args.device) if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    writer = SummaryWriter()

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
    summary(model=model, input_size=(2, 3, 224, 224))
    model.to(device=device)

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
    for epoch in range(args.max_epoch):
        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
        )

        # validate
        val_loss, val_acc = evaluate(
            model=model, data_loader=val_loader, device=device, epoch=epoch
        )

        tags = [
            "Loss/train",
            "Loss/test",
            "Accuracy/train",
            "Accuracy/test",
            "Learning_rate",
        ]
        writer.add_scalar(tags[0], train_loss, epoch)
        writer.add_scalar(tags[1], val_loss, epoch)
        writer.add_scalar(tags[2], train_acc, epoch)
        writer.add_scalar(tags[3], val_acc, epoch)
        writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        writer.close()

        if os.path.exists("./checkpoints/save_weights") is False:
            os.makedirs("./checkpoints/save_weights")
        if best_acc < val_acc:
            torch.save(model.state_dict(), "./checkpoints/save_weights/best_model.pth")
            best_acc = val_acc
        if epoch > (args.max_epoch - 5):
            torch.save(
                model.state_dict(), f"./checkpoints/save_weights/last_model-{epoch}.pth"
            )


if __name__ == "__main__":

    opt = DefaultConfig()

    main(opt)
