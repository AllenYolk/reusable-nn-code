import abc
import os


class TaskPipeline(abc.ABC):

    def __init__(self, imp, log_dir: str):
        self.imp = imp
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log_dir = log_dir

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test(self, *args, **kwargs):
        pass


class SupervisedTaskPipeline(TaskPipeline):

    def __init__(self, net, log_dir: str, backend: str = "torch", **kwargs):
        if backend == "torch":
            from reunn.implementation import torch_imp
            imp = torch_imp.TorchPipelineImp(net=net, **kwargs)
        else:
            raise ValueError(f"{backend} backend not supported!")

        super().__init__(imp, log_dir)

    def train(
        self, epochs: int, validation: bool = False,
        rec_best_checkpoint: bool = False, rec_latest_checkpoint: bool = False,
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

            if validation_loss < min_loss:
                min_loss = validation_loss
                if rec_best_checkpoint:
                    self.imp.save_pipeline_state(
                        dir=os.path.join(self.dir, "best_checkpoint.pt"), 
                        validation_loss=validation_loss, 
                        trained_epoch=epoch
                    )

            if rec_latest_checkpoint:
                self.imp.save_pipeline_state(
                    dir=os.path.join(self.log_dir, "latest_checkpoint.pt"),
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

            if validation_acc > max_acc:
                max_acc = validation_acc
                if rec_best_checkpoint:
                    self.imp.save_pipeline_state(
                        dir=os.path.join(self.log_dir, "best_checkpoint.pt"), 
                        validation_loss=validation_loss, 
                        validation_acc=validation_acc,
                        trained_epoch=epoch
                    )

            if rec_latest_checkpoint:
                self.imp.save_pipeline_state(
                    dir=os.path.join(self.log_dir, "latest_checkpoint.pt"),
                    validation_loss=validation_loss,
                    validation_acc=validation_acc,
                    trained_epoch=epoch
                )

    def test(self):
        test_loss, test_acc = self.imp.test_step(compute_acc=True)
        print(f"test_loss={test_loss}, test_acc={test_acc}")
