

class ConvModelEvaluator():
    def __init__(self, model, data, checkpoint_path):
        self.model = model.load_weigths(checkpoint_path)
        self.data = data




