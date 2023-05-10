import os
import shutil
import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience(default=7)."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path=".pth", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf  # np.Inf: positive infinite
        self.val_acc_max = 0
        self.current_epoch = 0  # initialized current epoch

    def __call__(self, val_loss, val_acc, model, current_epoch, hyperparameter):
        self.current_epoch = current_epoch
        score = -val_loss  # 这样做是为了让delta可以取正数

        if self.best_score is None:  # 第一次：将score直接传给best_score
            self.best_score = score
            self._save_checkpoint(val_loss, val_acc, model, hyperparameter)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:  # print msg when saving models to checkpoints
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, val_acc, model, hyperparameter)
            self.counter = 0

    def _save_checkpoint(self, val_loss, val_acc, model, arg_dirname: str):
        """save model when validation loss decrease."""
        # create a directory under current working directory ready for all the checkpoints
        relative_path = arg_dirname
        absolute_path = os.path.abspath(
            os.path.join(os.getcwd(), "checkpoints", relative_path)
        )  # get data root path

        if not (os.path.exists(absolute_path) and os.path.isdir(absolute_path)):
            os.mkdir(absolute_path)

        self.trace_func(
            f"val_loss decreased ({self.val_loss_min:.6f} ==> {val_loss:.6f})\n"
            f"val_acc increased ({self.val_acc_max:.6f} ==> {val_acc:.6f})"
        )

        #         for file in os.listdir(absolute_path):  # delete all inferior models
        #             path = os.path.join(absolute_path, file)
        #             try:
        #                 shutil.rmtree(path)
        #             except OSError:
        #                 os.remove(path)
        loss_epoch_notion = (
            f"last_model-{self.current_epoch}-val_acc-{val_acc:.4f}-{self.path}"
        )
        save_path = os.path.join(absolute_path, loss_epoch_notion)
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
        self.val_acc_max = val_acc
