# Super Resolution

Undergraduate thesis on super resolution


## Training a model

To train a neural network, a configuration file (json) is used to determine various parameters for training. This is also handy in storing the information of each training job. An example configuration file is shown below

```json
{
    "model": "srresnet",
    "solver": "std",
    "epochs": 10,
    "loss_fn": "MSE",
    ["checkpoint": "path"],
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
    ["scheduler": {
        "name": "reduce",
        params: {

        }
    }]
}
```

The list of available of options for each parameter are

model - srresnet, srgan, esrgan, cincgan  
solver - std, gan, med
dataset - div2k, flickr2k, df2k, ms-coco, urban100  
optimizer - adam
