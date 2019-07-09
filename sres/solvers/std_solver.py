import os
import torch
from datetime import datetime
from .base_solver import BaseSolver

class StandardSolver(BaseSolver):
    def __init__(self, model, optimizer, loss_fn, dataloader, scheduler=None, checkpoint=None):
        super().__init__()
        self.use_cuda = not False and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint = checkpoint
        self.dataloader = dataloader
        self.logger = self._init_logger('gan_solver')
        self.loss_fn = loss_fn

    
    def solve(self, epochs, batch_size, logdir):
        date = datetime.today().strftime('%m_%d')
        if self.checkpoint:
            chkpt = torch.load(self.checkpoint)
            self.model.load_state_dict(chkpt['model_state_dict'])
            self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            start_epoch = chkpt['epoch']
            loss = chkpt['loss']

        self.logger.info('')
        self.logger.info('Batch Size : %d' % batch_size)
        self.logger.info('Number of Epochs : %d' % epochs)
        self.logger.info('Steps per Epoch : %d' % len(self.dataloader))
        self.logger.info('')

        self.model.train()
        start_epoch = start_epoch if self.checkpoint else 0
        best_loss = 1e8
        for epoch in range(start_epoch, epochs):
            self.logger.info('============== Epoch %d/%d ==============' % (epoch+1, epochs))
            mean_loss = 0
            for step, image_pair in enumerate(self.dataloader):
                lres_img, hres_img = image_pair
                lres_img.to(self.device); hres_img.to(self.device)

                generated_img = self.model(lres_img)
                self.optimizer.zero_grad()
                loss = self.loss_fn(generated_img, hres_img)
                loss.backward()
                self.optimizer.step()

                self.logger.info('step: %d, loss: %.3f' % (step, loss.item()))
                mean_loss += loss.item()
    
            self.logger.info('epoch : %d, average loss : %.3f' % (epoch+1, mean_loss/len(self.dataloader)))

            if self.scheduler:
                self.scheduler.step()
    
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_path = '%s_checkpoint_%d_%s%s' % (self.model.name, epoch+1, date, '.pt')
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    f = save_path)
                self.logger.info('Checkpoint saved to %s' % save_path)

        self.logger.info('Training Complete')