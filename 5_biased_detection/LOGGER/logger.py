import torch
import torch.utils.tensorboard as tensorboard
from torchvision.utils import make_grid

class Logger:
    def __init__(self, log_dir):
        self.writer = tensorboard.SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, images):
        normalized_images = (images - images.min()) / (images.max() - images.min())
        grid_images = make_grid((normalized_images * 255).to(torch.uint8), nrow=8, normalize=False)
        self.writer.add_image(tag, grid_images)

    def log_histogram(self, tag, classifier):
        last_fc_layer = classifier.mlp[-1]
        weights = last_fc_layer.weight
        self.writer.add_histogram(tag, weights)

    def log_embeddings(self, tag, embeddings, metadata, step):
        self.writer.add_embedding(embeddings, metadata=metadata, global_step=step, tag=tag)

    def close(self):
        self.writer.close()