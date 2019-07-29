import json

class Config:
    def __init__(self, path):
        """ 
        args:
            path (str): path to the configuration file
        """
        with open(path, mode='r') as f:
            conf = json.load(f)
        
        self.conf = conf
        self.logdir = conf['logdir']
        self.model = conf['model']
        self.epochs = conf['epochs']
        self.loss_fn = conf['loss_fn']
        self.solver = conf['solver']
        self.patch_size = conf['patch_size']
        
        ds = conf['dataset']
        self.dataset_root = ds['root']
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
        
        if 'generator_path' in conf:
            self.generator_path = conf['generator_path']
        else:
            self.generator_path = None

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __str__(self):
        return str(self.conf)