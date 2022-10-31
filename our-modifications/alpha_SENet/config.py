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
        "alpha_2_0_resnext50_32x4d",
    ]  # 使用的模型，名字必须与models/__init__.py中的名字一致

    batch_size = 32
    use_gpu = True
    num_workers = 8
    device = "cuda:4"
    num_classes = 34
    resolution = 224  # Different Resolution are: [128, 160, 224, 320, 384]
    time_stamp = time.strftime("%m_%d-%H_%M_%S")

    # train config
    debug_mode = True
    max_epoch = 100
    warmup_epochs = 5
    learning_rate = (
        1e-1 * batch_size / 256
    )  # `bag-of-tricks`: warmup-to-MAX(0.1 * batch_size / 256), then COSINE-DECAY-TO-ZERO
    weight_decay = 1e-4  # `ResNet-RS impl.`: DEFAULT=1e-4, decrease to 4e-5 when using more regularization

    # test config
    load_model_path = [
        "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/seresnext50_32x4d-10_28-23_21_31/best_model.pth",
        "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/alpha_resnet50-10_28-21_41_10/best_model.pth",
        "/home/tr/myproject/XHGNet/our-modifications/alpha_SENet/checkpoints/alpha_resnext50_32x4d-10_28-22_31_37/best_model.pth",
    ]

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device(self.device) if opt.use_gpu else t.device("cpu")

        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("_"):
                print(k, getattr(self, k))


opt = DefaultConfig()
