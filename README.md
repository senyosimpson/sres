# Super Resolution

Undergraduate thesis on super resolution

## Training a model

To train a model, a json configuration file must be specified to determine various options for training. The configuration file is saved to the log directory when a job is run. An example configuration file is shown below

```json
{
    "model": "srresnet",
    "solver": "std",
    "epochs": 10,
    "loss_fn": "mse",
    "checkpoint": "path",
    "dataset": {
        "name": "div2k",
        "params": {
            "batch_size": 16,
            "num_workers": 2,
            "shuffle": true
        }
    },
    "optimizer": {
        "name": "adam",
        "params": {
            "lr" : 1e-4,
            "betas": [0.9, 0.99]
        }
    },
    "scheduler": {
        "name": "reduce",
        "params": {

        }
    }
}
```

The list of available of options for each parameter are
```
model - srresnet, srgan, esrgan 
solver - std, gan
dataset - div2k, flickr2k, df2k
optimizer - adam
loss_fn - mse
scheduler - reduce, cyclic
```

To Note:
 * checkpoint and scheduler are optional.
 * loss_fn may be a list if multiple loss_fns are used

In order to train a model, a solver must be defined for it. A solver is a class that contains the logic for training the model and takes in various arguments in order to do so. These are defined in the `solvers` directory. Every solver should inherit the base solver which sets defaults for every model. The current implementation is shown below 

```python
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

    def _init_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    @abstractmethod
    def solve(self, epochs, batch_size, logdir):
        raise NotImplementedError
```

The `solve` method must be implemented and is the method called to train the network. Additionally, the `__init__` method must take in the model as input in the derived class. An example

```python
from .base_solver import BaseSolver

class StandardSolver(BaseSolver):
    def __init__(self, model, optimizer, loss_fn, dataloader, scheduler=None, checkpoint=None):
        super().__init__(optimizer, loss_fn, dataloader, scheduler, checkpoint)
        self.model = model
        self.logger = self._init_logger('std_solver')

    
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
                save_path = '%s_res_checkpoint_%s%s' % (self.model.name, date, '.pt')
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    f = save_path)
                self.logger.info('Checkpoint saved to %s' % save_path)

        self.logger.info('Training Complete')
```

## Adding Options to Configuration

To add extra options to a configuration file, they must be specified in the `constants.py` file. This file defines a dictionary for every option (e.g losses, optimizer). Each key in the option links to a class, functions, variable that is instatiated in the `train.py` code at run time.

To get an overview of the current options (as some of the names are not self-explanatory), look in the `constants.py` file.