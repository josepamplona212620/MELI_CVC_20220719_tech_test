class BaseTrain(object):
    def __init__(self, model, data, config):
        self.model = model
        self.train_data = data[0]
        self.test_data = data[1]
        self.config = config

    def train(self):
        raise NotImplementedError
