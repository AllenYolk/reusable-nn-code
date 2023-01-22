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
    def validation_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save_pipeline_state(self, dir: str):
        pass

    @abc.abstractmethod
    def load_pipeline_state(self, dir: str):
        pass

    @abc.abstractmethod
    def add_runtime_records(self, main_tag, kv, idx):
        pass

    @abc.abstractmethod
    def clear_runtime_records(self):
        pass
