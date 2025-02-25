import torch
import logging
from abc import ABC, abstractmethod


class BaseSolver(ABC):
    """ Base class for solver objects. Defines an interface
    for all derived solvers to follow.

    Every derived class must implement the load_checkpoint and solve
    methods which are used for loading checkpoints and defining the logic
    for training a model respectively.
    """
    def __init__(self, conf, optimizer, loss_fn, dataloader, scheduler=None):
        """
        args:
            conf (Config): specified config used to train model
            optimizer (torch.Optim): optimizer to use for training 
                (must be instantiated)
            loss_fn ():
            dataloader (): pytorch dataloader object (must be instantiated)
            scheduler (): pytorch scheduling object (must be instantiated)
        """
        super().__init__()
        self.use_cuda = not False and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.start_epoch = 0
        self.best_loss = 0
        self.conf = conf.conf

    def _init_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def save_checkpoint(self, save_path, model_state_dict, opt_state_dict, conf, epoch, loss):
        """
        Saves a training checkpoint
        args:
            save_path (str): path to save checkpoint
            model_state_dict[s] (dict/list): model state dict[s] to save.
                if a list of model state dicts, expects a list of format
                [{name: modelA_state_dict}, {name: modelB_state_dict}] otherwise
                just pass in the normal state dict and given default name/key, model_state_dict
            opt_state_dict[s] (dict/list): same principle applies above. Default name/key given
                is optimizer_state_dict if a regular state dict is passed in
            epoch (int): the current epoch
            loss (torch.tensor): the current loss
        """
        info = {'epoch': epoch, 'loss': loss, 'config': conf}
        
        if isinstance(model_state_dict, list):
            for state_dict in model_state_dict:
                info.update(state_dict)
        else:
            info.update({'model_state_dict': model_state_dict})

        if isinstance(opt_state_dict, list):
            for state_dict in opt_state_dict:
                info.update(state_dict)
        else:
            info.update({'optimizer_state_dict': opt_state_dict})
        torch.save(info, f=save_path)
    
    @abstractmethod
    def load_checkpoint(self, checkpoint):
        raise NotImplementedError

    @abstractmethod
    def solve(self, epochs, batch_size, logdir, checkpoint=None):
        raise NotImplementedError