import os
import torch
from datetime import datetime
from .base_solver import BaseSolver

class GANSolver(BaseSolver):
    def __init__(self, conf, generator, discriminator, optimizers, loss_fns, dataloader, generator_path=None, scheduler=None):
        super().__init__(conf, optimizers, loss_fns, dataloader, scheduler)
        self.generator = generator
        if generator_path:
            self.load_generator(generator_path)
        self.discriminator = discriminator
        self.g_optimizer, self.d_optimizer = self.optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.logger = self._init_logger('gan_solver')
        self.g_loss_fn, self.d_loss_fn = self.loss_fn
        self.best_gen_loss = 1e8
        self.best_disc_loss = 1e8

    def load_generator(self, path):
        chkpt = torch.load(path)
        model_state_dict = chkpt['model_state_dict']
        self.generator.load_state_dict(model_state_dict)

    def load_checkpoint(self, checkpoint):
        chkpt = torch.load(self.checkpoint)
        self.generator.load_state_dict(chkpt['generator_state_dict'])
        self.discriminator.load_state_dict(chkpt['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(chkpt['optimizer_gen_state_dict'])
        self.d_optimizer.load_state_dict(chkpt['optimizer_disc_state_dict'])
        start_epoch = chkpt['epoch']
        self.best_gen_loss = chkpt['loss']['best_gen_loss']
        self.best_disc_loss = chkpt['loss']['best_disc_loss']
    
    def solve(self, epochs, batch_size, logdir, checkpoint=None):
        date = datetime.today().strftime('%m_%d')
        if checkpoint:
            self.load_checkpoint(checkpoint)

        self.logger.info('')
        self.logger.info('Batch Size : %d' % batch_size)
        self.logger.info('Number of Epochs : %d' % epochs)
        self.logger.info('Steps per Epoch : %d' % len(self.dataloader))
        self.logger.info('')

        self.generator.train()
        self.discriminator.train()
        start_epoch = self.start_epoch if checkpoint else 0
        best_gen_loss = self.best_gen_loss if checkpoint else 1e8
        best_disc_loss = self.best_disc_loss if checkpoint else 1e8
        for epoch in range(start_epoch, epochs):
            self.logger.info('============== Epoch %d/%d ==============' % (epoch+1, epochs))
            mean_gen_loss = 0
            mean_disc_loss = 0
            for step, image_pair in enumerate(self.dataloader):
                # discriminator has dense network which requires
                # batch size of 16 therefore skip the last set
                # of images
                if step == (len(self.dataloader)-1):
                    continue

                lres_img, hres_img = image_pair
                lres_img.to(self.device); hres_img.to(self.device)

                # train discriminator
                self.discriminator.zero_grad()
                generated_img = self.generator(lres_img)
                prediction_generated = self.discriminator(generated_img.detach())
                prediction_real = self.discriminator(hres_img)

                target_real = 0.8 + (torch.rand(batch_size, 1) * 0.2) # between 0.8 - 1.0
                target_gen = torch.rand(batch_size, 1) * 0.2 # between 0.0 - 0.2
                target_gen.to(self.device); target_real.to(self.device)

                d_loss_fake = self.d_loss_fn(prediction_generated, target_gen)
                d_loss_fake.backward()
                d_loss_real = self.d_loss_fn(prediction_real, target_real)
                d_loss_real.backward()
                d_loss = d_loss_real + d_loss_fake
                self.d_optimizer.step()

                # train generator
                self.generator.zero_grad()
                target_real = torch.ones(batch_size, 1).to(self.device)
                g_loss = self.g_loss_fn(generated_img, hres_img,
                    prediction_generated.detach(), target_real)
                g_loss.backward()
                self.g_optimizer.step()
                
                mean_gen_loss += g_loss.item()
                mean_disc_loss += d_loss.item()

                self.logger.info('Step: %d, Gen loss: %.5f, Discrim Loss: %.5f' % (step, g_loss.item(), d_loss.item()))

            if self.scheduler:
                self.scheduler.step()

            _gen_loss = mean_gen_loss / (len(self.dataloader) - 1)
            _disc_loss = mean_disc_loss / (len(self.dataloader) - 1)
            self.logger.info('epoch : %d, average gen loss : %.5f, average discrim loss : %.5f' % (epoch+1, _gen_loss, _disc_loss))

            if epoch % 10 == 0:
                best_gen_loss = mean_gen_loss
                best_disc_loss = mean_disc_loss
                save_path = '%s_checkpoint_%d_%s%s' % (self.generator.name, epoch+1, date, '.pt')
                save_path = os.path.join(logdir, save_path)
                model_state_dicts = [
                    {'generator_state_dict': self.generator.state_dict()},
                    {'discriminator_state_dict': self.discriminator.state_dict()}
                ]

                optimizer_state_dicts = [
                    {'optimizer_gen_state_dict': self.g_optimizer.state_dict()},
                    {'optimizer_disc_state_dict': self.d_optimizer.state_dict()}
                ]

                loss = {'best_gen_loss': best_gen_loss, 'best_disc_loss': best_disc_loss}
                self.save_checkpoint(save_path,
                                     model_state_dicts,
                                     optimizer_state_dicts,
                                     self.conf,
                                     epoch,
                                     loss)
                self.logger.info('Checkpoint saved to %s' % save_path)

        self.logger.info('Training Complete')