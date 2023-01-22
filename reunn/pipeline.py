import abc


class TaskPipeline(abc.ABC):

    def __init__(self, imp):
        self.imp = imp

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test(self, *args, **kwargs):
        pass


class SupervisedTaskPipeline(TaskPipeline):

    def __init__(self, net, backend: str = "torch", **kwargs):
        if backend == "torch":
            from reunn.implementation import torch_imp
            imp = torch_imp.TorchPipelineImp(net=net, **kwargs)
        else:
            raise ValueError(f"{backend} backend not supported!")

        super().__init__(imp)

    def train(self, epochs: int, validation: bool = False):
        for epoch in range(epochs):
            train_loss, _, validation_loss, _ = self.imp.train_step(validation)
            if validation:
                print(
                    f"epoch {epoch}: train_loss={train_loss}, "
                    f"validation_loss={validation_loss}"
                )
            else:
                print(f"epoch {epoch}: train_loss={train_loss}")

    def test(self):
        test_loss, _ = self.imp.test_step(compute_acc=False)
        print(f"test_loss={test_loss}")


class SupervisedClassificationTaskPipeline(SupervisedTaskPipeline):

    def __init__(self, net, backend: str = "torch", **kwargs):
        super().__init__(net, backend, **kwargs)

    def train(self, epochs: int, validation: bool = False):
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

    def test(self):
        test_loss, test_acc = self.imp.test_step(compute_acc=True)
        print(f"test_loss={test_loss}, test_acc={test_acc}")
