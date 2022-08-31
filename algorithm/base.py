import os


class Algorithm:

    def __init__(self,
                 content_dir,
                 style_dir,
                 epochs=1,
                 batch_size=1,
                 image_size=(256, 256),
                 mode='train'
                 ):

        self.model_name = self._set_class_name()

        # Dataset Parameters
        self.content_dir = content_dir
        self.style_dir = style_dir

        # Training Parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size

        self.mode = mode

    def evaluate(self):
        pass

    def train(self):
        pass

    def build_model(self):
        pass

    @classmethod
    def _set_class_name(cls):
        return cls.__name__

    @classmethod
    def _create_result_folder(cls):
        sub_folder_names = [
            'results',
            'checkpoints',
            'evaluates',
            'inferences',
            'logs'
        ]
        for names in sub_folder_names:
            os.makedirs(os.path.join(cls.__name__, names), exist_ok=True)
