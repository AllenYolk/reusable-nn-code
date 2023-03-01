import abc
from typing import Dict, List
from collections import defaultdict


class TaskPipeline(abc.ABC):

    def __init__(self, imp):
        self.imp = imp

    @abc.abstractmethod
    def train(self, *args, **kwargs) -> Dict:
        pass

    @abc.abstractmethod
    def test(self, *args, **kwargs) -> Dict:
        pass

    def save_pipeline_state(self, file: str, **kwargs):
        self.imp.save_pipeline_state(file=file, **kwargs)

    def load_pipeline_state(self, file: str):
        chk = self.imp.load_pipeline_state(file=file)
        exclude_k = [
            "net_state_dict", "optimizer_state_dict", "lr_scheduler_state_dict"
        ]
        for k, v in chk.items():
            if k in exclude_k:
                continue
            print(f"{k}={v}")
        return chk

    def add_runtime_records(self, main_tag, kv, idx):
        self.imp.add_runtime_records(main_tag, kv, idx)

    def add_hparam_records(self, metrics: dict):
        self.imp.add_hparam_records(metrics)


class SupervisedTaskPipeline(TaskPipeline):

    def __init__(
        self, net, log_dir: str, hparam: dict, backend: str = "torch", 
        **kwargs
    ):
        if backend == "torch":
            from reunn.implementation import torch_imp
            imp = torch_imp.TorchPipelineImp(
                net=net, log_dir=log_dir, hparam=hparam, **kwargs
            )
        elif backend == "spikingjelly":
            from reunn.implementation import spikingjelly_imp
            imp = spikingjelly_imp.SpikingjellyActivationBasedPipelineImp(
                net=net, log_dir=log_dir, hparam=hparam, **kwargs
            )
        else:
            raise ValueError(f"{backend} backend not supported!")
        super().__init__(imp)

    def train(
        self, epochs: int, validation: bool = False,
        rec_best_checkpoint: bool = False, rec_latest_checkpoint: bool = False,
        rec_runtime_msg: bool = False, rec_hparam_msg: bool = False, 
        silent: bool = False,
    ) -> Dict[str, List[float]]:
        min_loss = float("inf")
        results = defaultdict(list)

        for epoch in range(epochs):
            train_loss, _, validation_loss, _ = self.imp.train_step(
                validation, silent=silent
            )
            results["train_loss"].append(train_loss)
            if validation:
                results["validation_loss"].append(validation_loss)

            if not silent:
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

            min_loss_type = "validation" if validation else "train"
            if validation:
                if validation_loss < min_loss:
                    min_loss = validation_loss
                    if rec_best_checkpoint: 
                        self.imp.save_pipeline_state(
                            file="best_checkpoint.pt", 
                            validation_loss=validation_loss, 
                            train_loss=train_loss,
                            min_loss_type=min_loss_type,
                            trained_epoch=epoch
                        )
            else:
                if train_loss < min_loss:
                    min_loss = train_loss
                    if rec_best_checkpoint:
                        self.imp.save_pipeline_state(
                            file="best_checkpoint.pt", 
                            train_loss=train_loss, 
                            min_loss_type=min_loss_type,
                            trained_epoch=epoch
                        )

            if rec_latest_checkpoint:
                self.imp.save_pipeline_state(
                    file="latest_checkpoint.pt",
                    validation_loss=validation_loss,
                    train_loss=train_loss,
                    min_loss_type=min_loss_type,
                    trained_epoch=epoch
                )

        print(f"Training finished! min_{min_loss_type}_loss={min_loss}")
        if rec_hparam_msg:
            self.imp.add_hparam_records(
                metrics={f"min_{min_loss_type}_loss": min_loss}
            )
        return results

    def test(self, rec_hparam_msg: bool = False) -> Dict[str, float]:
        test_loss, _ = self.imp.test_step(compute_acc=False)
        print(f"Test finished! test_loss={test_loss}")
        if rec_hparam_msg:
            self.imp.add_hparam_records(
                metrics={"test_loss": test_loss}
            )
        return {"test_loss": test_loss}


class SupervisedClassificationTaskPipeline(SupervisedTaskPipeline):

    def __init__(
        self, net, log_dir: str, hparam: dict, backend: str = "torch", 
        **kwargs
    ):
        super().__init__(net, log_dir, hparam, backend, **kwargs)

    def train(
        self, epochs: int, validation: bool = False, 
        rec_best_checkpoint: bool = False, rec_latest_checkpoint: bool = False,
        rec_runtime_msg: bool = False, rec_hparam_msg: bool = False,
        silent: bool = False
    ) -> Dict[str, List[float]]:
        max_acc = -1.
        results = defaultdict(list)

        for epoch in range(epochs):
            train_loss, train_acc, validation_loss, validation_acc =\
                self.imp.train_step(
                    validation, compute_acc=True, silent=silent
                )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            if validation:
                results["validation_loss"].append(validation_loss)
                results["validation_acc"].append(validation_acc)

            if not silent:
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

            max_acc_type = "validation" if validation else "train"
            if validation:
                if validation_acc > max_acc:
                    max_acc = validation_acc
                    if rec_best_checkpoint:
                        self.imp.save_pipeline_state(
                            file="best_checkpoint.pt", 
                            validation_loss=validation_loss, 
                            validation_acc=validation_acc,
                            train_loss=train_loss,
                            train_acc=train_acc,
                            max_acc_type=max_acc_type,
                            trained_epoch=epoch
                        )
            else:
                if train_acc > max_acc:
                    max_acc = train_acc
                    if rec_best_checkpoint:
                        self.imp.save_pipeline_state(
                            file="best_checkpoint.pt",
                            train_loss=train_loss,
                            train_acc=train_acc,
                            max_acc_type=max_acc_type,
                            trained_epoch=epoch
                        )

            if rec_latest_checkpoint:
                self.imp.save_pipeline_state(
                    file="latest_checkpoint.pt",
                    validation_loss=validation_loss,
                    validation_acc=validation_acc,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    max_acc_type=max_acc_type,
                    trained_epoch=epoch
                )

        print(f"Training finished! max_{max_acc_type}_acc={max_acc}")
        if rec_hparam_msg:
            self.imp.add_hparam_records(
                metrics={f"max_{max_acc_type}_acc": max_acc}
            )
        return results

    def test(self, rec_hparam_msg: bool = False) -> Dict[str, float]:
        test_loss, test_acc = self.imp.test_step(compute_acc=True)
        print(f"Test finished! test_loss={test_loss}, test_acc={test_acc}")
        if rec_hparam_msg:
            self.imp.rec_hparam_msg(
                metrics={"test_acc": test_acc, "test_loss": test_loss}
            )
        return {"test_loss": test_loss, "test_acc": test_acc}
