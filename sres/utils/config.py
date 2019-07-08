import json

class Config:
    def __init__(self, path):
        """ 
        args:
            path (str): path to the configuration file
        """
        with open(path, mode='r') as f:
            conf = json.load(f)
        
        self.model_name = conf['model']
        self.epochs = conf['epochs']
        self.loss_fn = conf['loss_fn']
        
        ds = conf['dataset']
        self.dataset_name = ds['name']
        self.dataset_params = ds['params']

        opt = conf['optimizer']
        self.optimizer_name = opt['name']
        self.optimizer_parms = opt['params']

        if 'checkpoint' in conf:
            self.checkpoint = conf['checkpoint']
        else:
            self.checkpoint = None