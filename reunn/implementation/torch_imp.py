from reunn.implementation import base_imp


class TorchPipelineImp(base_imp.BasePipelineImp):

    def __init__(self):
        super().__init__()

    def train_step(self):
        return "train"

    def test_step(self):
        return "test"