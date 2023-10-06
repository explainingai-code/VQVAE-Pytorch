VQVAE Implementation in pytorch with generation using LSTM
========

This repository implements a VQVAE for mnist and colored version of mnist and follows up with a simple LSTM for generating numbers.

Video on VQVAE - https://www.youtube.com/watch?v=1ZHzAOutcnw

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/VQVAE-Pytorch.git```
* ```cd VQVAE-Pytorch```
* ```pip install -r requirements.txt```
* For running a simple VQVAE with minimal code to understand the basics ```python run_simple_vqvae.py```
* For playing around with VQVAE and training/inferencing the LSTM use the below commands passing the desired configuration file as the config argument 
* ```python -m tools.train_vqvae``` for training vqvae
* ```python -m tools.infer_vqvae``` for generating reconstructions and encoder outputs for LSTM training
* ```python -m tools.train_lstm``` for training minimal LSTM 
* ```python -m tools.generate_images``` for using the trained LSTM to generate some numbers

## Configurations
* ```config/vqvae_mnist.yaml``` - VQVAE for training on black and white mnist images
* ```config/vqvae_colored_mnist.yaml``` - VQVAE with more embedding vectors for training colored mnist images 

## Data preparation
For setting up the dataset:
Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Verify the data directory has the following structure:
```
VQVAE-Pytorch/data/train/images/{0/1/.../9}
	*.png
VQVAE-Pytorch/data/test/images/{0/1/.../9}
	*.png
```

## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created and ```output_train_dir``` will be created inside it.

During training of VQVAE the following output will be saved 
* Best Model checkpoints(VQVAE and LSTM) in ```task_name``` directory

During inference the following output will be saved
* Reconstructions for sample of test set in ```task_name/output_train_dir/reconstruction.png``` 
* Encoder outputs on train set for LSTM training in ```task_name/output_train_dir/mnist_encodings.pkl```
* LSTM generation output in ```task_name/output_train_dir/generation_results.png```


## Sample Output for VQVAE

Running `run_simple_vqvae` should give you below reconstructions

Running default config VQVAE for mnist should give you below reconstructions for both versions

Sample Generation Output after just 5 epochs

Training the lstm longer and more parameters will give better results 
