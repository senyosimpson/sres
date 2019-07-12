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
    "patch_size": 128,
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
 * loss_fn may be a list if multiple loss functions are used

In order to train a model, a solver must be defined for it. A solver is a class that contains the logic for training the model and takes in various arguments in order to do so. These are defined in the `solvers` directory. Every solver should inherit the base solver which sets defaults for every model. The current implementation is shown below 

```python
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self, optimizer, loss_fn, dataloader, scheduler=None):
            super().__init__()
            self.use_cuda = not False and torch.cuda.is_available()
            self.device = torch.device('cuda' if self.use_cuda else 'cpu')
            self.optimizer = optimizer
            self.scheduler = scheduler
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

    def save_checkpoint(self, save_path, model_state_dict, opt_state_dict, epoch, loss):
        """
        Saves a training checkpoint
        args:
            save_path (str): path to save checkpoint
            model_state_dict[s] (dict/list): model state dict[s] to save.
                if a list of model state dicts, expects a list of format
                [{name: modelA_state_dict}, {name: modelB_state_dict}] otherwise
                just pass in the normal state dict and given default name/key,
                model_state_dict
            opt_state_dict[s] (dict/list): same principle applies above. Default 
            name/key given is optimizer_state_dict if a regular state dict is passed in
            epoch (int): the current epoch
            loss (torch.tensor): the current loss
        """
        info = {'epoch': epoch, 'loss': loss}
        
        if isinstance(model_state_dict, list):
            for state_dict in model_state_dict:
                info.update(state_dict)
        else:
            info.update({'model_state_dict', model_state_dict})

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
```

The `solve` method must be implemented and is the method called to train the network. Additionally, the `__init__` method must take in the model as input in the derived class. An example

```python
from .base_solver import BaseSolver

class StandardSolver(BaseSolver):
    def __init__(self, model, optimizer, loss_fn, dataloader, scheduler=None):
        super().__init__(optimizer, loss_fn, dataloader, scheduler)
        self.model = model
        self.logger = self._init_logger('std_solver')

    def load_checkpoint(self, checkpoint):
        chkpt = torch.load(checkpoint)
        self.model.load_state_dict(chkpt['model_state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        self.start_epoch = chkpt['epoch']
    
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
                save_path = os.path.join(logdir, save_path)
                self.save_checkpoint(save_path,
                                     self.model.state_dict(),
                                     self.optimizer.state_dict(),
                                     epoch,
                                     loss)
                self.logger.info('Checkpoint saved to %s' % save_path)

        self.logger.info('Training Complete')
```

## Adding Options to Configuration

To add extra options to a configuration file, they must be specified in the `constants.py` file. This file defines a dictionary for every option (e.g losses, optimizer). Each key in the option links to a class, functions, variable that is instatiated in the `train.py` code at run time.

To get an overview of the current options (as some of the names are not self-explanatory), look in the `constants.py` file.

## Paperspace

To train models, the service used was [Paperspace](https://www.paperspace.com). The script `submit_train_job.sh` is used to submit a job to paperspace. It requires three environment variables to be set, `CONTAINER_NAME, DOCKERHUB_USERNAME, DOCKERHUB_PASSWORD`. The dockerhub username and password are necessary because the repository used is private. If it is public, edit the script accordingly.

In order to use it, a docker container must be created and hosted on a platform such as [DockerHub](https://hub.docker.com). The script, `Dockerfile` can be used to build this docker container. If you are using DockerHub, run the commands

```bash
docker build -t <hub-user>/<repo-name>[:tag] .
docker push <hub-user>/<repo-name>[:tag]
```

This will create a docker container and push it to the required repository. This docker container is then utilized by the script used to submit a training job.