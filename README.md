VQVAE Implementation in pytorch with generation using LSTM
========

This repository implements [VQVAE](https://arxiv.org/abs/1711.00937) for mnist and colored version of mnist and follows up with a simple LSTM for generating numbers.

## VQVAE Explanation and Implementation Video
<a href="https://www.youtube.com/watch?v=1ZHzAOutcnw">
   <img alt="VQVAE Video" src="https://github.com/explainingai-code/VQVAE-Pytorch/assets/144267687/a411d732-8c99-41fb-b39c-dd2c3fbfa448"
   width="300">
</a>


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

Running `run_simple_vqvae` should be very quick (as its very simple model) and give you below reconstructions (input in black black background and reconstruction in white background)

<img src="https://github.com/explainingai-code/VQVAE-Pytorch/assets/144267687/607fb5a8-b880-4af5-8ce0-5d7127aa66a7" width="400">

Running default config VQVAE for mnist should give you below reconstructions for both versions

<img src="https://github.com/explainingai-code/VQVAE-Pytorch/assets/144267687/939f8f22-0145-467f-8cd6-4b6c6e6f315f" width="400">
<img src="https://github.com/explainingai-code/VQVAE-Pytorch/assets/144267687/0e28286a-bc4c-44e3-a385-84d1ae99492c" width="400">

Sample Generation Output after just 10 epochs
Training the vqvae and lstm longer and more parameters(codebook size, codebook dimension, channels , lstm hidden dimension e.t.c) will give better results 

<img src="https://github.com/explainingai-code/VQVAE-Pytorch/assets/144267687/688a6631-df34-4fde-9508-a05ae3c2ae91" width="300">
<img src="https://github.com/explainingai-code/VQVAE-Pytorch/assets/144267687/187fa630-a7ef-4f0b-aef7-5c6b53019b38" width="300">

## Citations
```
@misc{oord2018neural,
      title={Neural Discrete Representation Learning}, 
      author={Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu},
      year={2018},
      eprint={1711.00937},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


