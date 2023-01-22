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

    def train(
        self, epochs: int, validation: bool = False, compute_acc: bool = False,
    ):
        for epoch in range(epochs):
            self.imp.train_step(validation, compute_acc)

    def test(self, compute_acc: bool = False):
        self.imp.test_step(compute_acc)
