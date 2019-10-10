# Diary for Super Resolution Thesis

## 11/07/19

* Trained SRResNet from scratch
* Dataset used was df2k and loss was MSE
* Loss stopped improving at around 78 epochs
* Iterations were approximately 13k
* Plan is to resume training but implementing a reduce learning rate on plateau schedule

## 12/07/19

* Training SRResNet from checkpoint.
* Using reduce lr on plateau schedule

* Loss still not decreasing
* Loss may not be a good indicator of learning progress
* Idea is to track PSNR and watch if that increases over time

## 14/07/19

* Evaluated PSNR on SRResNet model
* Average PSNR of set5 is approximately 29.16 dB

## 24/07/19

* Trained SRGAN model (with correct code)
* Results are still poor
* Used incorrect mean, std for the vgg network
* Idea is to only normalize images when they go through the vgg network

## 25/07/19

* Realized generated image and real image are never compared directly, only through VGG map
* VGG expects normalized images of a certain standard
* Reason why colours are off when generated
* Started training job for SRResNet with normalized inputs
* Idea is to train GAN after that

## 27/07/19

* Normalized the images and trained SRResNet
* Loss stalled at around 0.05
* PSNR of Set5 is only 28.5 dB
* Plan is to only train the network on DIV2K for some fine-tuning

* Fine-tuning did not seem to be improving loss
* Started training a GAN using the new pretrained SRResNet

## 28/07/19

* SRGAN has a checkerboard artifact
* Colour is still somewhat off
* Implemented a new feature extractor that normalizes input image in VGGLoss
* Added weighted MSELoss to Perceptual Loss function (to handle checkerboard artifacts)
* Removed normalization from training code
* Idea is to train a GAN using SRResNet.pt (instead of new normalized one)

* Launched training job for SRGAN using updated code

## 29/07/19

* Heavy checkerboard artifacts but colour mapping is still okay
* However, only trained for 20 epochs which is roughly 4600 iterations
* This is orders of magnitude less than the SOTA (roughly 100 000) iterations
* Need to train for at least 20 000 epochs to see if the approach is valid
* Confident the implementation is right

## 17/08/19

* Output layer of generator has a tanh activation function
* Still using FC network in discriminator but maybe should use FCN
* Training job up and running
* Created a modified SRResNet model based on the one find in xinntao repo
* Training job should be launched after current SRGAN model

## 18/08/19

* Checkerboarding gone!
* Tanh activation is crucial
* Network arbitrarily trains at double the speed (must have to do something with the activation function)
* Launching job for MSRResNet
* Average PSNR is 28.7453 dB for set5 (lower than other network)
* Launching a job without tanh activation
* After that, try one with bilinear interpolation instead of bicubic

* Next step is to train a network on only the Y channel
* In inference pipeline, crop boundary of 4 on each side as boundary effects decrease results

## 19/09/19

* MSRResNet got a PSNR of 29.4 (without tanh layer)
* Happy with that
* Can add both SRResNet and MSRResNet in my model evaluation for thesis
* Next step is to train a GAN with the tanh layer and without

## 20/09/19

* Trained GAN without tanh layer, results are positive
* Went to 30 epochs
* PSNR is ~ 28.5 dB which is reasonable
* No checkerboarding with MSRResNet
* Might be useful to take out batch normalization in the discriminator as well
* Once functionality to log images to /storage, launch job
* Ready to launch long haul training job, ~ 2 to 3 days