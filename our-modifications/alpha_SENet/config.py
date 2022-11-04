import time
import warnings
import torch as t
from data_config.constants import SMALL_XHGNET_DEFAULT_MEAN, SMALL_XHGNET_DEFAULT_STD


class DefaultConfig(object):

    train_data_root = "../../data/small-XHGNet/train/"
    test_data_root = "../../data/small-XHGNet/val/"
    data_mean = SMALL_XHGNET_DEFAULT_MEAN
    data_std = SMALL_XHGNET_DEFAULT_STD

    # commen config
    model = [
        "densenet161",
        # "resnext50_32x4d",
        # "alpha_1_1_resnext50_32x4d",
        # "alpha_2_0_resnext50_32x4d",
        # "seresnext50_32x4d",
        # "resnext101_32x4d",
        # "alpha_1_1_resnext101_32x4d",
        # "alpha_2_0_resnext101_32x4d",
        # "seresnext101_32x4d",
        # "resnext101_32x8d",
        # "alpha_1_1_resnext101_32x8d",
        # "alpha_2_0_resnext101_32x8d",
        # "seresnext101_32x8d",
        # "resnext101_64x4d",
        # "alpha_1_1_resnext101_64x4d",
        # "alpha_2_0_resnext101_64x4d",
        # "seresnext101_64x4d",
    ]  # 使用的模型，名字必须与models/__init__.py中的名字一致

    # 0 for "cpu", 1 for "single_gpu", 2 for "multi_gpu"
    gpu_mode = 1
    if gpu_mode == 0:
        device = t.device("cpu")
    elif gpu_mode == 1:
        device = t.device("cuda:6")
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
    num_classes = 34
    resolution = 224  # Different Resolution are: [128, 160, 224, 320, 384]
    time_stamp = time.strftime("%m_%d-%H_%M_%S")

    # train config
    debug_mode = True
    max_epoch = 80
    learning_rate = (
        1e-1 * batch_size / 256
    )  # `bag-of-tricks`: warmup-to-MAX(0.1 * batch_size / 256), then COSINE-DECAY-TO-ZERO
    weight_decay = 1e-4  # `ResNet-RS impl.`: DEFAULT=1e-4, decrease to 4e-5 when using more regularization
    use_lr_scheduler = True
    if use_lr_scheduler:
        warmup_epochs = 5

    # test config
    load_model_path = [
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/resnext101_32x4d-10_31-10_09_11/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/alpha_1_1_resnext101_32x4d-10_31-10_09_11/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/alpha_2_0_resnext101_32x4d-10_31-10_09_11/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/seresnext101_32x4d-10_31-10_09_11/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/resnext101_32x8d-10_31-10_13_06/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/alpha_1_1_resnext101_32x8d-10_31-10_13_06/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/alpha_2_0_resnext101_32x8d-10_31-10_13_06/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/seresnext101_32x8d-10_31-10_13_06/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/resnext101_64x4d-10_31-10_15_15/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/alpha_1_1_resnext101_64x4d-10_31-10_15_15/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/alpha_2_0_resnext101_64x4d-10_31-10_15_15/best_model.pth",
        # "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/seresnext101_64x4d-10_31-10_15_15/best_model.pth",
    ]

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
