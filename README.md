# Thesis

This repository is the official repository for my Master Thesis titled Stochastic Undirected Neural Networks.
More information about the work developed can be found in the documents "Thesis.pdf" and "Extended_Abstract.pdf" (to be added after the 24th of November).

## Content

The folder "Code" contains the code of the project. 
The folder "Models" contains the weights of the models used in this project

The following example shows how to train and test a SUNN model with CNN operations

# SUNN application example

The first step is to log in into W&B platform and create a new project. Let's consider a project titled mnist_cnn_example, for training SUNN models with cnn operations.

![create_project](https://github.com/ricardosimoes00/Thesis/assets/93200673/c256c8f7-f4c8-4cc4-84a1-bcba238a1cb5)

Then, inside the new project, we create a new sweep and select the desired ranges and values for the different hyperparameters, like seen below

![hyperparameters_example](https://github.com/ricardosimoes00/Thesis/assets/93200673/db47b590-17ed-413e-a281-36a82bcd4e3d)

Here is an example of a hyperparameter list configuration:

```
method: random
parameters:
  activation:
    values:
      - relu
      - tanh
  backward_loss_coef:
    distribution: uniform
    max: 1
    min: 0.01
  batch_size:
    values:
      - 128
      - 256
      - 512
  constrained:
    values:
      - "True"
  device:
    values:
      - cpu
  dropout:
    distribution: uniform
    max: 0.5
    min: 0.1
  lr:
    distribution: uniform
    max: 0.01
    min: 0.0005
  max_epochs:
    values:
      - 70
  noise_mean:
    values:
      - 0
  noise_sd:
    distribution: uniform
    max: 1
    min: 0
  seed:
    values:
      - 42
  unn_iter:
    values:
      - 1
      - 3
      - 5
      - 10
  unn_y_init:
    values:
      - uniform
      - rand
      - zero
program: mnist_conv_stochastic.py
```

Note that the "program" line must contain the name of the model we will use, which in this case is the SUNN with CNN operations


After this, we just need to obatin the API key, which can be found in https://wandb.ai/settings and the name of the sweep we just created.

Training is initiated by running the command below, replacing [user name] by the user name of the account
and [sweep name] by the name of the sweep. I recommend using Google Colab since it provides a free GPU could environment, useful for training DL models.

```
!wandb agent [user name]/mnist_cnn/[sweep name]
```

When using Google Colab, all the weights can be saved locally using the following command:

```
shutil.make_archive("/content/wandb", 'zip', "/content/wandb")
files.download("/content/wandb.zip")
```

After selecting a particular configurations and a specific epoch, we can select that model for test time, obtaining predictions and evaluation metrics, or even the generation of prototypes.

Refer to the following Notebook for an example: https://colab.research.google.com/drive/1lsnzG66McNaHUL3ZGaj7BNLhrhZxuyxp?usp=sharing
