import time
import warnings
import torch as t
from data_config.constants import XHGNET_V4_MEAN, XHGNET_V4_STD


class DefaultConfig(object):
    train_data_root = "../../data/XHGNetV4/train/"
    test_data_root = "../../data/XHGNetV4/val/"
    data_mean = XHGNET_V4_MEAN
    data_std = XHGNET_V4_STD
    val_rate = 0.2

    # commen config
    model = [
        # "sse_rd101_ada_resnext50dd_32x4d",
        # "sse_rd101_ada_resnext101_32x4d",
        # "sse_rd101_ada_resnext101_32x8d",
        # "sse_rd101_ada_resnext101_64x4d",
        # "efficientnetv2_s",
        # "efficientnetv2_m",
        # "efficientnetv2_l",
        # "convnext_tiny",
        # "convnext_small",
        # "convnext_base",
        # "regnety_016",
        # "regnety_032",
        # "regnety_040",
        # "regnety_064",
        # "regnety_080",
        # "regnety_120",
        # "convnext_tiny",
        # "convnext_small",
        # "convnext_base",
        # "resnet50",
        # "ecamresnet50",
        # "cbamresnet50",
        # "sse_rd101_ada_resnet50dd",
        # "sse_rd102_ada_resnet152dd",
        # "sse_rd104_ada_resnet152dd",
        # "sse_rd108_ada_resnet152dd",
        # "sse_rd116_ada_resnet152dd",
        # "sse_rd132_ada_resnet152dd",
        # "mobilenetv3_large_075",
        "mobilenetv3_large_100",
    ]  # 使用的模型，名字必须与models/__init__.py中的名字一致
    pretrain = False
    continue_training = False
    use_earlystop = True
    earlystop_patience = 12

    # 0 for "cpu", 1 for "single_gpu", 2 for "multi_gpu"
    gpu_mode = 1
    if gpu_mode == 0:
        device = t.device("cpu")
    elif gpu_mode == 1:
        device = t.device("cuda:4")
    else:
        # TODO: update DDP training script
        device = t.device("cuda")
        use_gpus = gpu_mode
        rank = None
        world_size = None
        gpu = None
        distributed = None
        dist_backend = None
        dist_url = "env://"

    base_bs = 32
    batch_size = base_bs * use_gpus if gpu_mode > 1 else base_bs

    num_workers = 8
    num_classes = 11
    resolution = 224  # Different Resolution are: [128, 160, 224, 320, 384]
    time_stamp = time.strftime("%m_%d-%H_%M_%S")

    # train config
    max_epoch = 100
    optimizer = "AdamW"
    debug_mode = True

    # for AdamW, const. LR
    # 0.00004, 0.001, 0.0001, 0.0005
    learning_rate = 0.0005  # `bag-of-tricks`: for SGD, warmup-to-MAX(0.1 * batch_size / 256), then COSINE-DECAY-TO-ZERO
    weight_decay = 1e-5  # `ResNet-RS impl.`: DEFAULT=1e-4, decrease to 4e-5 when using more regularization

    use_lr_scheduler = False
    if use_lr_scheduler:
        warmup_epochs = 5

    # test config
    load_model_path = [
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_27-22_06_55-sse_rd101_ada_resnext50dd_32x4d-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_27-22_20_30-sse_rd101_ada_resnext101_32x4d-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_27-22_24_16-sse_rd101_ada_resnext101_32x8d-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_27-22_27_37-sse_rd101_ada_resnext101_64x4d-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_27-23_09_16-efficientnetv2_s-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_27-23_12_31-efficientnetv2_m-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_27-23_16_56-efficientnetv2_l-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_27-23_15_12-no-atten_efficientnetv2_l-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/qualitative_analysis_checkpoints/resnet50.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/qualitative_analysis_checkpoints/ecamresnet50.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/qualitative_analysis_checkpoints/cbamresnet50.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/qualitative_analysis_checkpoints/sse_rd101_ada_resnet50dd.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-00_08_30-convnext_tiny-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-00_08_30-convnext_small-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-00_08_30-convnext_base-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-08_28_33-regnety_016-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-08_28_33-regnety_032-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-08_28_33-regnety_040-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-08_28_33-regnety_064-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-08_28_33-regnety_080-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/05_28-08_28_33-regnety_120-LR_False_0.0005-BS_32-WD_1e-05/best_model.pth",
    ]

    # inference config
    # CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM, XGradCAM, LayerCAM
    cam_method = "GradCAMpp"

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        if kwargs is not None:
            for k, v in kwargs.items():
                if not hasattr(self, k):
                    warnings.warn("Warning: opt has not attribut %s" % k)
                setattr(self, k, v)

        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("_"):
                print(k, getattr(self, k))


opt = DefaultConfig()


if __name__ == "__main__":
    opt._parse(kwargs=None)
