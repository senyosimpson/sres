import torch
import logging
from abc import ABC, abstractmethod


class BaseSolver:
    def __init__(self, optimizer, loss_fn, dataloader, scheduler=None, checkpoint=None):
            super().__init__()
            self.use_cuda = not False and torch.cuda.is_available()
            self.device = torch.device('cuda' if self.use_cuda else 'cpu')
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.checkpoint = checkpoint
            self.dataloader = dataloader
            self.loss_fn = loss_fn
            self.start_epoch = 0

    def _init_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    @abstractmethod
    def save_checkpoint(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_checkpoint(self, checkpoint):
        raise NotImplementedError

    @abstractmethod
    def solve(self, epochs, batch_size, logdir):
        raise NotImplementedError