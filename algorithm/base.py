import os


class Algorithm:

    def __init__(self, content_dir, style_dir, epochs=1, batch_size=1, image_size=(256, 256)):
        # Dataset Parameters
        self.content_dir = content_dir
        self.style_dir = style_dir

        # Training Parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size

        self._create_result_folder()

    def evaluate(self):
        pass

    def train(self):
        pass

    def build_model(self):
        pass

    @classmethod
    def _create_result_folder(cls):
        sub_folder_names = [
            'model_results',
            'model_checkpoints'
        ]
        for names in sub_folder_names:
            os.makedirs(os.path.join(cls.__name__, names), exist_ok=True)
