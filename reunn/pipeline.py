import abc
from typing import Dict


class TaskPipeline(abc.ABC):

    def __init__(self, imp):
        self.imp = imp

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test(self, *args, **kwargs):
        pass

    def save_pipeline_state(self, file: str, extra_value_dict: Dict):
        self.imp.save_pipeline_state(file=file, **extra_value_dict)

    def load_pipeline_state(self, file: str):
        chk = self.imp.load_pipeline_state(file=file)
        for k, v in chk.items():
            if k in ["state_dict"]:
                continue
            print(f"{k}={v}")
        return chk

    def add_runtime_records(self, main_tag, kv, idx):
        self.imp.add_runtime_records(main_tag, kv, idx)

    def clear_runtime_records(self):
        self.imp.clear_runtime_records()


class SupervisedTaskPipeline(TaskPipeline):

    def __init__(self, net, log_dir: str, backend: str = "torch", **kwargs):
        if backend == "torch":
            from reunn.implementation import torch_imp
            imp = torch_imp.TorchPipelineImp(net=net, log_dir=log_dir, **kwargs)
        else:
            raise ValueError(f"{backend} backend not supported!")
        super().__init__(imp)

    def train(
        self, epochs: int, validation: bool = False,
        rec_best_checkpoint: bool = False, rec_latest_checkpoint: bool = False,
        rec_runtime_msg: bool = False,
    ):
        min_loss = float("inf")
        for epoch in range(epochs):
            train_loss, _, validation_loss, _ = self.imp.train_step(validation)

            if validation:
                print(
                    f"epoch {epoch}: train_loss={train_loss}, "
                    f"validation_loss={validation_loss}"
                )
            else:
                print(f"epoch {epoch}: train_loss={train_loss}")

            if rec_runtime_msg:
                kv = {"train_loss": train_loss}
                if validation:
                    kv["validation_loss"] = validation_loss
                self.add_runtime_records(main_tag="loss", kv=kv, idx=epoch)

            if validation_loss < min_loss:
                min_loss = validation_loss
                if rec_best_checkpoint:
                    self.imp.save_pipeline_state(
                        file="best_checkpoint.pt", 
                        validation_loss=validation_loss, 
                        trained_epoch=epoch
                    )

            if rec_latest_checkpoint:
                self.imp.save_pipeline_state(
                    file="latest_checkpoint.pt",
                    validation_loss=validation_loss,
                    trained_epoch=epoch
                )

        print(f"Training finished! min_validation_loss={min_loss}")

    def test(self):
        test_loss, _ = self.imp.test_step(compute_acc=False)
        print(f"test_loss={test_loss}")


class SupervisedClassificationTaskPipeline(SupervisedTaskPipeline):

    def __init__(self, net, log_dir, backend: str = "torch", **kwargs):
        super().__init__(net, log_dir, backend, **kwargs)

    def train(
        self, epochs: int, validation: bool = False, 
        rec_best_checkpoint: bool = False, rec_latest_checkpoint: bool = False,
        rec_runtime_msg: bool = False,
    ):
        max_acc = -1.
        for epoch in range(epochs):
            train_loss, train_acc, validation_loss, validation_acc =\
                self.imp.train_step(validation, compute_acc=True)

            if validation:
                print(
                    f"epoch {epoch}: train_loss={train_loss}, "
                    f"train_acc={train_acc}, "
                    f"validation_loss={validation_loss}, "
                    f"validation_acc={validation_acc}"
                )
            else:
                print(
                    f"epoch {epoch}: train_loss={train_loss}, "
                    f"train_acc={train_acc}"
                )

            if rec_runtime_msg:
                kv_loss = {"train_loss": train_loss}
                kv_acc = {"train_acc": train_acc}
                if validation:
                    kv_loss["validation_loss"] = validation_loss
                    kv_acc["validation_acc"] = validation_acc
                self.add_runtime_records(main_tag="loss", kv=kv_loss, idx=epoch)
                self.add_runtime_records(main_tag="acc", kv=kv_acc, idx=epoch)

            if validation_acc > max_acc:
                max_acc = validation_acc
                if rec_best_checkpoint:
                    self.imp.save_pipeline_state(
                        file="best_checkpoint.pt", 
                        validation_loss=validation_loss, 
                        validation_acc=validation_acc,
                        trained_epoch=epoch
                    )

            if rec_latest_checkpoint:
                self.imp.save_pipeline_state(
                    file="latest_checkpoint.pt",
                    validation_loss=validation_loss,
                    validation_acc=validation_acc,
                    trained_epoch=epoch
                )

    def test(self):
        test_loss, test_acc = self.imp.test_step(compute_acc=True)
        print(f"test_loss={test_loss}, test_acc={test_acc}")
