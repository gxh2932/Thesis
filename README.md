# Thesis
Code for Master's Thesis.
[Click here to download thesis.](https://github.com/gxh2932/Thesis/files/10496450/Thesis.5.pdf) Results from the experiments performed can be found in the **results** folder.

## Organization
The provided modules serve the following purpose:

* **main.py**: Provides functions for training models with different layers.

* **layer_defs.py**: Contains definitions for different normalization layers. 

* **models.py**: Contains definitions for different CNN model architectures.

* **config.py**: Training hyperparameters and progress bar definition.

* **run_class_similarity.py**: Compute cosine similarity between class features.

* **utils.py**: Contains helper functions for run_class_similarity.py.

* **run_hessian.py**: Compute properties of the Hessian matrix.

* **transformer_model.py**: Contains definitions for different Vision Transformer model architectures.

## Example execution 
To train a model (e.g. VGG) using a particular normalization layer (e.g. BatchNorm), run the following command

```execution
python main.py -arch=ResNet-56 --norm_type=BatchNorm
```

## Summary of basic options

```--arch=<architecture> ```

- *Options*: vgg / resnet-56 / vit. 

```--p_grouping=<amount_of_grouping_in_GroupNorm> ```

- *Options*: integer; *default*: 32. 
- If p_grouping < 1: defines a group size of 1/p_grouping. E.g., p_grouping=0.5 implies group size of 2. 
- If p_grouping >= 1: defines number of groups as layer_width/p_grouping. E.g., p_grouping=32 implies number of groups per layer will be 32.

```--skipinit=<use_skipinit_initialization> ```

- *Options*: True/False; *Default*: False. 

```--preact=<use_preactivation_resnet> ```

- *Options*: True/False; *Default*: False. 

```--probe_layers=<probe_activations_and_gradients> ```

- *Options*: True/False; *Default*: True
- Different properties in model layers (activation norm, stable rank, std. dev., cosine similarity, and gradient norm) will be calculated every iteration and stored as a dict every 5 epochs of training

```--init_lr=<init_lr> ```

- *Options*: float; *Default*: 1. 
- A multiplication factor to alter the learning rate schedule (e.g., if default learning rate is 0.1, init_lr=0.1 will make initial learning rate be equal to 0.01).

```--lr_warmup=<lr_warmup> ```

- *Options*: True/False; *Default*: False.
- Learning rate warmup; used in Filter Response Normalization.

```--batch_size=<batch_size> ```

- *Options*: integer; *Default*: 256. 

```--dataset=<dataset> ```

- *Options*: CIFAR-10/CIFAR-100; *Default*: CIFAR-100.

```--download=<download_dataset> ```

- *Options*: True/False; *Default*: False.
- If CIFAR-10 or CIFAR-100 are to be downloaded, this option should be True.

```--cfg=<number_of_layers> ```

- *Options*: cfg_10/cfg_20/cfg_40; *Default*: cfg_10
- Number of layers for VGG architectures.

```--seed=<change_random_seed> ```

- *Options*: integer; *Default*: 0.

**Training Settings**: To change number of epochs or the learning rate schedule for training, change the hyperparameters in *config.py*. By default, models are trained using SGD with momentum (0.9).
