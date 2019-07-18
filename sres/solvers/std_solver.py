import os
import torch
from datetime import datetime
from .base_solver import BaseSolver

class StandardSolver(BaseSolver):
    def __init__(self, conf, model, optimizer, loss_fn, dataloader, scheduler=None):
        super().__init__(conf, optimizer, loss_fn, dataloader, scheduler)
        self.model = model
        self.logger = self._init_logger('std_solver')

    def load_checkpoint(self, checkpoint):
        chkpt = torch.load(checkpoint)
        self.model.load_state_dict(chkpt['model_state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        self.start_epoch = chkpt['epoch']
        self.best_loss = chkpt['loss']
    
    def solve(self, epochs, batch_size, logdir, checkpoint=None):
        date = datetime.today().strftime('%m_%d')
        if checkpoint:
            self.load_checkpoint(checkpoint)

        self.logger.info('')
        self.logger.info('Batch Size : %d' % batch_size)
        self.logger.info('Number of Epochs : %d' % epochs)
        self.logger.info('Steps per Epoch : %d' % len(self.dataloader))
        self.logger.info('')

        self.model.train()
        start_epoch = self.start_epoch if checkpoint else 0
        best_loss = self.best_loss if checkpoint else 1e8
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
                self.scheduler.step(mean_loss)
    
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_path = '%s_res_checkpoint_%s%s' % (self.model.name, date, '.pt')
                save_path = os.path.join(logdir, save_path)
                self.save_checkpoint(save_path,
                                     self.model.state_dict(),
                                     self.optimizer.state_dict(),
                                     self.conf,
                                     epoch,
                                     loss)
                self.logger.info('Checkpoint saved to %s' % save_path)

        self.logger.info('Training Complete')