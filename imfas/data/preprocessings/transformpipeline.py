import torch.nn as nn


class TransformPipeline(nn.Sequential):
    def __init__(self, modulelist: list):
        super(TransformPipeline, self).__init__(*modulelist)
        self.fitted = False

    def fit(self, input):
        """
        first time around fit & transform on training data. In the
        second run, simply use the nn.Sequentials forward method (is also its call
        to transform a test set to the very same scale!
        """
        for module in self:
            input = module.fit(input)

        self.fitted = True
        return input

    def transform(self, input):
        for module in self:
            input = module.transform(input)
        return input
