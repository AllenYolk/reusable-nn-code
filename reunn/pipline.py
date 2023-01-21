class TaskPipeline:

    def __init__(self, imp):
        self._imp = imp


class SupervisedTaskPipeline(TaskPipeline):

    def __init__(self, imp, backend: str = "torch"):
        super().__init__(imp)
        pass
