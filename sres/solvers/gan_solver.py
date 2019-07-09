import os
from datetime import datetime
from .base_solver import BaseSolver

class GANSolver(BaseSolver):
    def __init__(self, generator, discriminator, optimizers, losses, dataloader, scheduler=None, checkpoint=None):
        super().__init__()
        self.use_cuda = not False and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.generator = generator
        self.discriminator = discriminator
        self.generator.to(self.device); self.discriminator.to(self.device)
        self.g_optimizer, self.d_optimizer = optimizers
        self.scheduler = scheduler
        self.checkpoint = checkpoint
        self.dataloader = dataloader
        self.logger = self._init_logger('gan_solver')
        self.g_loss, self.d_loss = losses

    
    def solve(self, epochs, batch_size, logdir):
        date = datetime.today().strftime('%m_%d')
        if self.checkpoint:
            chkpt = torch.load(self.checkpoint)
            self.generator.load_state_dict(chkpt['generator_state_dict'])
            self.discriminator.load_state_dict(chkpt['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(chkpt['optimizer_gen_state_dict'])
            self.d_optimizer.load_state_dict(chkpt['optimizer_disc_state_dict'])
            start_epoch = chkpt['epoch']
            loss = chkpt['loss']

        self.logger.info('')
        self.logger.info('Batch Size : %d' % batch_size)
        self.logger.info('Number of Epochs : %d' % epochs)
        self.logger.info('Steps per Epoch : %d' % len(self.dataloader))
        self.logger.info('')

        self.generator.train()
        self.discriminator.train()
        start_epoch = start_epoch if start_epoch else 0
        best_loss = 0
        for epoch in range(start_epoch, epochs):
            self.logger.info('============== Epoch %d/%d ==============' % (epoch+1, epochs))
            mean_loss = 0
            for step, image_pair in enumerate(self.dataloader):
                lres_img, hres_img = image_pair
                lres_img.to(self.device); hres_img.to(self.device)

                generated_img = self.generator(lres_img)
                prediction_generated = self.discriminator(generated_img)
                prediction_real = self.discriminator(hres_img)

                # train discriminator
                self.d_optimizer.zero_grad()
                target_gen = 0.7 + (torch.rand(batch_size, 1) * 0.5) # between 0.7 - 1.2
                target_real = torch.rand(batch_size, 1) * 0.3 # between 0.0 - 0.3

                target_gen.to(self.device); target_real.to(self.device)
                d_loss = self.d_loss(prediction_generated, target_gen) + self.d_loss(prediction_real, target_real)
                d_loss.backwards()
                self.d_optimizer.step()

                # train generator
                self.g_optimizer.zero_grad()
                g_loss = self.g_loss(generated_img, hres_img, prediction_generated)
                g_loss.backwards()
                self.g_optimizer.step()

                self.logger.info('Step: %d, Gen loss: %.3f, Discrim Loss: %.3f' % (step, g_loss.item(), d_loss.item()))

            if self.scheduler:
                self.scheduler.step()
    
            self.logger.info('epoch : %d, average loss : %.3f' % (epoch+1, 3.333))

            if mean_loss < best_loss:
                best_loss = mean_loss
                save_path = '%s_checkpoint_%d_%s%s' % (self.generator.name, epoch+1, date, '.pt')
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_gen_state_dict': self.g_optimizer.state_dict(),
                    'optimizer_disc_state_dict': self.d_optimizer.state_dict()},
                    f = os.path.join(logdir, save_path))
                self.logger.info('Checkpoint saved to %s' % save_path)

        self.logger.info('Training Complete')