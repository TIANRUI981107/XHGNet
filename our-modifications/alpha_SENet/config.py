import warnings
import torch as t


class DefaultConfig(object):

    train_data_root = "../../data/small-XHGNet/train/"  # 训练集存放路径
    test_data_root = "../../data/small-XHGNet/val/"  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    model = "resnet50"  # 使用的模型，名字必须与models/__init__.py中的名字一致

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 8  # how many workers for loading data
    device = "cuda:4"

    debug_file = "/tmp/debug"  # if os.path.exists(debug_file): enter ipdb
    result_file = "result.csv"

    # Training Configurations
    num_classes = 34
    max_epoch = 100
    warmup_epochs = 5
    learning_rate = (
        1e-1 / batch_size
    )  # ResNet-RS=warmup-to-MAX(0.1/BS), then COSINE-DECAY-TO-ZERO
    weight_decay = (
        0  # DEFAULT=1e-4, decrease to ResNet-RS=4e-5 when using more regularization
    )
    resolution = 224  # Different Resolution are: [128, 160, 224, 320]

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device("cuda:4") if opt.use_gpu else t.device("cpu")

        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("_"):
                print(k, getattr(self, k))


if __name__ == "__main__":
    opt = DefaultConfig()
