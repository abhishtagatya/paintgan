import csv


class CSVLogger:

    def __init__(self, model_name, style_dir, epochs, checkpoint_per=1):
        self.model_name = model_name
        self.checkpoint_per = checkpoint_per
        self.style = style_dir
        self.epochs = epochs

        self.filename = f'{self.model_name}/logs/{self.model_name}_{self.style}-{self.epochs}.csv'

    def compile(self, data):

        header = [
            'epoch', 'content_loss', 'style_loss', 'total_loss'
        ]

        with open(self.filename, 'w') as log_file:
            writer = csv.writer(log_file, delimiter=';')
            writer.writerow(header)
            writer.writerows(data)