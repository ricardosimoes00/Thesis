# Thesis

This repository is the official repository for my Master Thesis titled Stochastic Undirected Neural Networks.
More information about the work developed can be found in the documents "Tese.pdf" and "Extended_Abstract.pdf".

## Content

The folder "Code" contains the code of the project. 
The folder "Models" contains the weights of the models used in this project

The following example shows how to train and test a SUNN model with CNN operations

# 1st Step - Log in and create a new project

The first step is to log in into W&B platform and create a new project. Let's consider a project titled mnist_cnn_example, for training SUNN models with cnn operations.

![create_project](https://github.com/ricardosimoes00/Thesis/Example_Images/create_project.png)

Then, inside the new project, we create a new sweep and select the desired ranges and values for the different hyperparameters, like seen below

![hyperparameters_example](https://github.com/ricardosimoes00/Thesis/Example_Images/hyperparameters_example.png)

```
!wandb agent [user name]/mnist_cnn/[sweep name]
```

This instruction can be found in W&B platform, as seen below

![example](https://github.com/ricardosimoes00/Thesis/Example_Images/example.png)

