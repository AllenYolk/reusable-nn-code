import abc


class BasePipelineImp(abc.ABC):

    def __init__(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def acc_cnt(self, pred, labels):
        pass

    @abc.abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validate_step(self, *args, **kwargs):
        pass
