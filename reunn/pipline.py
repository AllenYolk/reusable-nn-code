class TaskPipeline:

    def __init__(self, imp):
        self.imp = imp

    def train(self):
        pass

    def test(self):
        pass


class SupervisedTaskPipeline(TaskPipeline):

    def __init__(self, backend: str = "torch", **kwargs):
        if backend == "torch":
            from reunn.implementation import torch_imp
            imp = torch_imp.TorchPipelineImp(**kwargs)
        else:
            raise ValueError(f"{backend} backend not supported!")

        super().__init__(imp)
