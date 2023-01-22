import abc


class BasePipelineImp(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def train_step(self):
        pass

    @abc.abstractmethod
    def test_step(self):
        pass
