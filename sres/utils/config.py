import json

class Config:
    def __init__(self, path):
        """ 
        args:
            path (str): path to the configuration file
        """
        with open(path, mode='r') as f:
            conf = json.load(f)
        
        self.model = conf['model']
        self.epochs = conf['epochs']
        self.loss_fn = conf['loss_fn']
        self.solver = conf['solver']
        
        ds = conf['dataset']
        self.dataset_name = ds['name']
        self.dataset_params = ds['params']

        opt = conf['optimizer']
        self.optimizer_name = opt['name']
        self.optimizer_params = opt['params']

        if 'scheduler' in conf:
            sched = conf['scheduler']
            self.scheduler_name = sched['name']
            self.scheduler_params = sched['params']
        else:
            self.scheduler_name = None
            self.scheduler_params = None

        if 'checkpoint' in conf:
            self.checkpoint = conf['checkpoint']
        else:
            self.checkpoint = None

    def __getitem__(self, key):
        return getattr(self, key)