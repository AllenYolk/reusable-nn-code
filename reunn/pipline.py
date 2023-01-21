class TaskPipeline:

    def __init__(self, imp):
        self._imp = imp


class SupervisedTaskPipeline(TaskPipeline):

    def __init__(self, backend: str = "torch"):
        pass