import math
from functools import partial
from dataclasses import dataclass
from typing import Optional
from omegaconf import OmegaConf
import torch
from torch.nn.modules import activation
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.datasets import MNIST
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
import random
import numpy as np
import os
import itertools

def reg_softmax(k): 

  k_copy = k.clone()
  k_flat = k_copy.view(k_copy.size(0), -1)
  k_flat[k_flat == 0] = 1
  shannon_entropy = - torch.sum(((k_flat)) * torch.log(((k_flat))), dim=1)
  # shannon_entropy = shannon_entropy.view(1, -1)
  
  return - shannon_entropy

def reg_relu(k):
  # print("k.shape:", k.shape)
  k_copy = k.clone()
  k_flat = k_copy.view(k_copy.size(0), -1)
  euclidean_norm = torch.norm(k_flat, dim=1)
  euclidean_norm = euclidean_norm.view(1, -1)
  
  return 0.5 * torch.square(euclidean_norm)

def reg_tanh(k):
  result_tensor = (k + 1) / 2 * torch.log((k + 1) / 2) + (k - 1) / 2 * torch.log((k - 1) / 2)
  sum_tensor = torch.sum(result_tensor, dim = (1,2,3))  # shape: [batch_size]
  return sum_tensor

def frobenius_inner_product(k1, k2):
    return torch.sum(k1 * k2, dim = (1,2,3))

# def load_config(conf, show=False):
#     # conf = OmegaConf.from_cli()

#     # validate against schema
#     schema = OmegaConf.structured(MNISTConvConfigSchema)
#     conf = OmegaConf.merge(schema, conf)

#     if show:
#         print(OmegaConf.to_yaml(conf))

#     conf = OmegaConf.to_container(conf)

#     return conf

def get_x_y(batch, cuda):
    # *_, w, h = batch[0].shape
    # images = batch[0].view(-1, w * h)  # between [0, 1].
    images = batch[0]
    images = 2*images - 1  # [between -1 and 1]
    labels = batch[1]

    if cuda:
        images = images.to("cuda")
        labels = labels.to("cuda")

    return images, labels


class ConvUNN(torch.nn.Module):


    def __init__(self, k, n_classes, activation, dropout_p=0, y_init="zero", seed = 42):
        super().__init__()
        self.k = k
        self.n_classes = n_classes
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.y_init = y_init
        self.seed = seed
        self.activation = activation
      

        # fixed arch. lazy

        d0 = 28
        d1 = 6
        n1 = 32
        d2 = 4
        n2 = 64
        d3 = 5
        n3 = n_classes

        self.stride = 2

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

        self.h1_dim = (self.d0 - self.d1) // self.stride + 1
        self.h2_dim = (self.h1_dim - self.d2) // self.stride + 1

        self.W1 = torch.nn.Parameter(torch.empty(n1, 1, d1, d1))
        self.b1 = torch.nn.Parameter(torch.empty(n1))

        self.W2 = torch.nn.Parameter(torch.empty(n2, n1, d2, d2))
        self.b2 = torch.nn.Parameter(torch.empty(n2))

        self.W3 = torch.nn.Parameter(torch.empty(n3, n2 * self.h2_dim * self.h2_dim))
        self.b3 = torch.nn.Parameter(torch.empty(n3))

        print("self.W1", self.W1.shape)
        print("self.b1", self.b1.shape)
        print("self.W2", self.W2.shape)
        print("self.b2", self.b2.shape)
        print("self.W3", self.W3.shape)
        print("self.b3", self.b3.shape)
        
        torch.manual_seed(self.seed)

        torch.nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
                                      # nonlinearity='tanh')
        torch.nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
                                      # nonlinearity='tanh')
        torch.nn.init.kaiming_uniform_(self.W3, a=math.sqrt(5))

        fan_in_1, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W1)
        bound_1 = 1 / math.sqrt(fan_in_1)
        torch.nn.init.uniform_(self.b1, -bound_1, bound_1)

        fan_in_2, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W2)
        bound_2 = 1 / math.sqrt(fan_in_2)
        torch.nn.init.uniform_(self.b2, -bound_2, bound_2)

        fan_in_3, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W3)
        bound_3 = 1 / math.sqrt(fan_in_3)
        torch.nn.init.uniform_(self.b3, -bound_3, bound_3)


    def _update_X(self, H1):
        return torch.conv_transpose2d(H1, weight=self.W1, stride=self.stride)

    def _update_H1(self, X, H2, b1_stochastic):

        H1_fwd = torch.conv2d(X, self.W1, self.b1 + b1_stochastic, stride=self.stride)
        H1_bwd = torch.conv_transpose2d(H2, weight=self.W2, stride=self.stride)
        return H1_fwd + H1_bwd

    def _update_H2(self, H1, Y, b2_stochastic):
        h1_dim = (self.d0 - self.d1) // self.stride + 1
        h2_dim = (h1_dim - self.d2) // self.stride + 1
        H2_fwd = torch.conv2d(H1, self.W2, self.b2 + b2_stochastic, stride=self.stride)
        H2_bwd = (Y @ self.W3).reshape(-1, self.n2, h2_dim, h2_dim)
        return H2_fwd + H2_bwd

    def _update_Y(self, H2, b3_stochastic):
        # flatten
        H2_ = H2.view(H2.shape[0], -1)
        return H2_ @ self.W3.T + (self.b3 + b3_stochastic)


    def forward(self, X, b1_stochastic, b2_stochastic, b3_stochastic):

        h1_dim = (self.d0 - self.d1) // self.stride + 1
        h2_dim = (h1_dim - self.d2) // self.stride + 1

        b = X.shape[0]
        H2 = torch.zeros((b, self.n2, h2_dim, h2_dim), device=X.device)
        
        # Initialize Y according to setup
        if self.y_init == "zero":
            # Initialize Y with zeros by default
            Y = torch.zeros(b, self.n3, device=X.device) 
        elif self.y_init == "rand":
            # Initialize Y as a random probability distribution
            Y = torch.rand(b, self.n3, device=X.device) 
            Y = torch.softmax(Y, dim=-1)
        elif self.y_init == "uniform":
            # Initialize Y as a random probability distribution
            Y = torch.zeros(b, self.n3, device=X.device) 
            Y = torch.softmax(Y, dim=-1)

        mask_H1 = torch.ones(b, self.n1, h1_dim, h1_dim, device=X.device)
        mask_H1 = self.dropout(mask_H1)

        mask_H2 = torch.ones(b, self.n2, h2_dim, h2_dim, device=X.device)
        mask_H2 = self.dropout(mask_H2)
        
        total_energy = []
        total_energy = []
        total_entropy = []
                       
        for i in range(self.k):
            if i==0: # Standard training of UNN with k steps of coordinate descent
                # for _ in range(self.k):

                X_flattened = torch.flatten(X, 1, 3)
                
                if torch.cuda.is_available(): 
                  device = torch.device("cuda")
                  X_flattened = X_flattened.to(device)
                  X = X.to(device)
                  Y = Y.to(device)
                  b1_stochastic = b1_stochastic.to(device)
                  H2 = H2.to(device)
                else:
                  device = torch.device("cpu")
                  X_flattened = X_flattened.to(device)
                  X = X.to(device)
                  Y = Y.to(device)
                  b1_stochastic = b1_stochastic.to(device)
                  H2 = H2.to(device)


                H1 = self._update_H1(X, H2, b1_stochastic)
                
                if self.activation == "relu":
                  H1 = torch.relu(H1)
                elif self.activation == "tanh":
                  H1 = torch.tanh(H1)
                
                if torch.cuda.is_available():
                  device = torch.device("cuda")
                  mask_H1 = mask_H1.to(device)
                  b2_stochastic = b2_stochastic.to(device)
                  H1 = H1.to(device)
                else:
                    device = torch.device("cpu")
                    mask_H1 = mask_H1.to(device)
                    b2_stochastic = b2_stochastic.to(device)
                    H1 = H1.to(device)

                H1 = H1 * mask_H1

                H2 = self._update_H2(H1, Y, b2_stochastic)

                if self.activation == "relu":
                  H2 = torch.relu(H2)
                elif self.activation == "tanh":
                  H2 = torch.tanh(H2)
                
                
                if torch.cuda.is_available():
                  device = torch.device("cuda")
                  mask_H2 = mask_H2.to(device)
                  b3_stochastic = b3_stochastic.to(device)
                  H2 = H2.to(device)
                else:
                  device = torch.device("cpu")
                  mask_H2 = mask_H2.to(device)
                  b3_stochastic = b3_stochastic.to(device)
                  H2 = H2.to(device)

                H2 = H2 * mask_H2

                Y_logits = self._update_Y(H2, b3_stochastic)
                Y = torch.softmax(Y_logits, dim=-1)
                
                H2 = self._update_H2(H1, Y, b2_stochastic)

                if self.activation == "relu":
                  H2 = torch.relu(H2)
                elif self.activation == "tanh":
                  H2 = torch.tanh(H2)
                
                H2 = H2 * mask_H2
                entropy = reg_softmax(Y)
                total_entropy.append(-torch.sum(entropy))
                ###### Energy Calculation´######

                E_X = 0.5 * (frobenius_inner_product(X, X)) #Parece ok
                if self.activation == "relu":
                  # print("entrei relu")
                  E_H1 = - torch.sum((((self.b1 + b1_stochastic).view(32, 1,1) * torch.ones((12,12), device = "cuda:0")).view(1, 32, 12, 12)*H1),dim = (1,2,3)) + reg_relu(H1)
                  E_H2 = - torch.sum((((self.b2 + b2_stochastic).view(64, 1,1) * torch.ones((5,5), device = "cuda:0")).view(1, 64, 5, 5)*H2),dim = (1,2,3)) + reg_relu(H2)
                elif self.activation == "tanh":
                  # print("entrei tanh")
                  E_H1 = - torch.sum((((self.b1 + b1_stochastic).view(32, 1,1) * torch.ones((12,12), device = "cuda:0")).view(1, 32, 12, 12)*H1),dim = (1,2,3)) + 0.5 * (frobenius_inner_product(X, X))
                  E_H2 = - torch.sum((((self.b2 + b2_stochastic).view(64, 1,1) * torch.ones((5,5), device = "cuda:0")).view(1, 64, 5, 5)*H2),dim = (1,2,3)) + 0.5 * (frobenius_inner_product(X, X))

                E_Y = - torch.sum(Y * (self.b3 + b3_stochastic).view(1, 10), dim=1) + reg_softmax(Y)

                E_XH1 = - torch.sum(torch.conv2d(X, self.W1, stride=self.stride) * H1, dim=(1, 2, 3))
                E_H1H2 = - torch.sum(torch.conv2d(H1, self.W2, stride=self.stride) * H2, dim=(1, 2, 3))

                energy = E_X + E_H1 + E_H2 + E_Y + E_XH1 + E_H1H2
                total_energy.append(torch.mean(energy).item())
               
                # print("b1_stochastic.shape:",b1_stochastic.shape)
                # print("b2_stochastic.shape:",b2_stochastic.shape)
                # print("b3_stochastic.shape:",b3_stochastic.shape)

                # print("self.b1.shape:",self.b1.shape)
                # print("self.b2.shape:",self.b2.shape)
                # print("self.b3.shape:",self.b3.shape)

                # print("E_X.shape:",E_X.shape)
                # print("E_H1.shape:",E_H1.shape)
                # print("E_H2.shape:",E_H2.shape)
                # print("E_Y.shape:",E_Y.shape)
                # print("E_XH1.shape:",E_XH1.shape)
                # print("E_H1H2.shape:",E_H1H2.shape)

                # print("torch.conv2d(X, self.W1, stride=self.stride).shape:",torch.conv2d(X, self.W1, stride=self.stride).shape)
                # print("H1.shape:",H1.shape)
                # print("torch.sum(conv_result * H1, dim=(1, 2, 3)):",torch.sum(torch.conv2d(X, self.W1, stride=self.stride) * H1, dim=(1, 2, 3)).shape)              
  
        # print(total_energy)                 
        return Y_logits, total_energy, total_entropy

    def backward(self, y, b1_stochastic, b2_stochastic, k=None, return_all_x=True):

        if k is None:
            k = self.k

        Y = torch.nn.functional.one_hot(y, num_classes=self.n_classes)
        b = Y.shape[0]

        all_X = []

        H1 = torch.zeros(b, self.n1, self.h1_dim, self.h1_dim, device=y.device)

        X = torch.zeros(b, 1, self.d0, self.d0, device=y.device)
        # X = torch.flatten(X, 1, 3)
        Y = Y.to(dtype=X.dtype)

        for i in range(k):
            H2 = self._update_H2(H1, Y, b2_stochastic)

            if self.activation == "relu":
              H2 = torch.relu(H2)
            elif self.activation == "tanh":
              H2 = torch.tanh(H2)

            H1 = self._update_H1(X, H2, b1_stochastic)
            
            if self.activation == "relu":
              H1 = torch.relu(H1)
            elif self.activation == "tanh":
              H1 = torch.tanh(H1)

            Xp = self._update_X(H1)
            if self.activation == "relu":
              X = torch.relu(Xp)
            elif self.activation == "tanh":
              X = torch.tanh(Xp)

            if return_all_x:
                all_X.append(X.detach().clone())

            H1 = self._update_H1(X, H2, b1_stochastic)
                
            if self.activation == "relu":
                H1 = torch.relu(H1)
            elif self.activation == "tanh":
                H1 = torch.tanh(H1)
                
            # if self.k > 1:
            #   X = torch.flatten(X, 1, 3)

        # returns pre-activation (logit) X as well as matrix of all Xs.
        return Xp, all_X

def main():
  
  run = wandb.init()
  conf = run.config

  backward_loss_coef = conf.backward_loss_coef
  batch_size = conf.batch_size
  device = conf.device
  dropout = conf.dropout
  lr = conf.lr
  max_epochs = conf.max_epochs
  noise_mean = conf.noise_mean
  noise_sd = conf.noise_sd
  seed = conf.seed
  unn_iter = conf.unn_iter
  unn_y_init = conf.unn_y_init
  activation = conf.activation
  
  
  if torch.cuda.is_available():
    device = "cuda"

    # print("config.dropout:",config.dropout)

    # conf = load_config(conf, show=True)
            
    # Step 1. Load Dataset
    train_and_dev_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

    n_train_and_dev = len(train_and_dev_dataset)
    n_dev = 10000
    train_dataset, dev_dataset = random_split(
        train_and_dev_dataset,
        [n_train_and_dev - n_dev, n_dev],
        generator=torch.Generator().manual_seed(42)
    )

    print("Train data", len(train_dataset))
    print("Dev   data", len(dev_dataset))
    print("Test  data", len(test_dataset))

    print("CONF:",conf)
    batch_size = conf['batch_size']
    _, w, h = train_dataset[0][0].shape
    input_dim = w * h
    output_dim = 10

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvUNN(k=conf['unn_iter'],
                      n_classes=output_dim,
                      dropout_p=conf['dropout'],
                      y_init=conf['unn_y_init'],	
                      seed = conf["seed"],
                      activation = conf["activation"]
                      )

    if torch.cuda.is_available():
        model = model.to("cuda")

    print(model)

    # optimizer = torch.optim.SGD(model.parameters(), lr=conf['lr'], momentum = 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])


    print('Num parameters:', sum(p.numel() for p in model.parameters()))

    # for p in model.parameters():
    #     print(p.numel())

    for name, param in model.named_parameters():
        print(name, param.numel())
        
    noise_sd = conf["noise_sd"]
    noise_mean = conf["noise_mean"]
    
    train_model(model,
                train_loader,
                dev_loader,
                test_loader,
                optimizer,
                conf,
                get_x_y,
                noise_sd,
                noise_mean)

    run.finish()


def train_model(model, train_loader, dev_loader, test_loader,
                optimizer, conf, get_x_y, noise_sd, noise_mean):

    n_train = len(train_loader.dataset)

    best_val_acc = 0
    best_val_acc_test_acc = None
    best_val_acc_epoch = None

    # accuracy_dev = eval_model(model, dev_loader, get_x_y)
    # accuracy_test = eval_model(model, test_loader, get_x_y)
    # print("Before training acc", accuracy_dev, accuracy_test)

    # computes softmax and then the cross entropy
    loss_fw = torch.nn.CrossEntropyLoss(reduction='none')
    loss_bw = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    epochs_list = []
    loss_fw_train_list = []
    loss_bw_train_list = []
    loss_fw_val_list = []
    loss_bw_val_list = []
    loss_fw_test_list = []
    loss_bw_test_list = []
    acc_train_list = []
    acc_val_list = []
    acc_test_list = []
    noise_seeds_list = []

    params_shape_list = []
    for p in model.parameters():
        print(p.numel())
        params_shape_list.append(p.numel())

    W1_shape, b1_shape, W2_shape, b2_shape, W3_shape, b3_shape = params_shape_list
    
    fw_increases = 0
    bw_increases = 0
    
    lowest_loss_fw_val = +np.inf
    lowest_loss_bw_val = +np.inf

    for epoch in range(conf['max_epochs']):

        # generate noise and keep track
        
        # random_int = torch.randint(low=1, high=101, size=(1,)).item()
        random_int = random.randint(1, 100)
        torch.manual_seed(random_int)
        b1_stochastic = torch.randn(b1_shape)*noise_sd + noise_mean
        b2_stochastic = torch.randn(b2_shape)*noise_sd + noise_mean
        b3_stochastic = torch.randn(b3_shape)*noise_sd + noise_mean

        if torch.cuda.is_available():
          b1_stochastic = b1_stochastic.to("cuda")
          b2_stochastic = b2_stochastic.to("cuda")
          b3_stochastic = b3_stochastic.to("cuda")

        noise_seeds_list.append(random_int)

        loss_fw_train = 0
        loss_bw_train = 0
        accuracy_train = 0

        loss_fw_val_total = 0 
        loss_bw_val_total = 0 

        for batch_id, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            if torch.cuda.is_available():
              cuda = torch.cuda.is_available()
            else:
              cuda = False

            x, y = get_x_y(batch, cuda)

            # [batch x n_classes]
            logits_fw, _, _aaa = model(x, b1_stochastic, b2_stochastic, b3_stochastic)
            # [batch]
            loss_val_fw = loss_fw(logits_fw, y)

            loss_avg = loss_val_fw.mean()
            if conf['backward_loss_coef'] > 0:

                # [batch x 1 x 28 x 28]
                logits_bw, _ = model.backward(y, b1_stochastic, b2_stochastic)
                # [batch x 1 x 28 x 28]
                loss_val_bw = loss_bw(logits_bw, (x>0).to(dtype=x.dtype))

                loss_avg = loss_avg + conf['backward_loss_coef'] * loss_val_bw.mean()

            loss_avg.backward()
            optimizer.step()

            loss_fw_train += loss_val_fw.sum().item()
            if conf['backward_loss_coef'] > 0:
                loss_bw_train += loss_val_bw.mean(dim=-1).sum().item()
            accuracy_train += (logits_fw.argmax(dim=1) == y).sum().item()

        accuracy_val, loss_fw_val, loss_bw_val  = eval_model(model, dev_loader, get_x_y, loss_fw, loss_bw, b1_stochastic, b2_stochastic, b3_stochastic, cuda)
        accuracy_test, loss_fw_test, loss_bw_test = eval_model(model, test_loader, get_x_y, loss_fw, loss_bw, b1_stochastic, b2_stochastic, b3_stochastic, cuda)

        loss_val_avg = loss_fw_val + conf["backward_loss_coef"]*loss_bw_val

        loss_fw_train /= n_train  # average sample loss
        loss_bw_train /= n_train  # average sample loss
        accuracy_train /= n_train
        
        # if accuracy_val > best_val_acc:
        #     best_val_acc = accuracy_val
        #     best_val_acc_test_acc = accuracy_test
        #     best_val_acc_epoch = epoch
            
        weights_dir = os.path.join(wandb.run.dir, wandb.run.name)
        os.makedirs(weights_dir, exist_ok=True)

        filename = f"{wandb.run.name}_epoch{epoch}.pt"
        torch.save(model, os.path.join(weights_dir, filename))

        log = {
            'epoch': epoch,
            'loss_fw_train': loss_fw_train,
            'loss_bw_train': loss_bw_train,
            'loss_fw_val:': loss_fw_val,
            'loss_bw_val:': loss_bw_val,
            'loss_fw_test:': loss_fw_test,
            'loss_bw_test:': loss_bw_test,
            'acc_train': accuracy_train,
            'acc_val': accuracy_val,
            'acc_test': accuracy_test,
            'noise_seed': random_int,
            "loss_val_avg": loss_val_avg
        }

        epochs_list.append(epoch)
        loss_fw_train_list.append(loss_fw_train)
        loss_bw_train_list.append(loss_bw_train)
        loss_fw_val_list.append(loss_fw_val)
        loss_bw_val_list.append(loss_bw_val)
        loss_fw_test_list.append(loss_fw_test)
        loss_bw_test_list.append(loss_bw_test)
        acc_train_list.append(accuracy_train)
        acc_val_list.append(accuracy_val)
        acc_test_list.append(accuracy_test)
        
        wandb.log(log)				
        print(log)

        # Stopping Criteria

        if loss_fw_val < lowest_loss_fw_val:
          lowest_loss_fw_val = loss_fw_val
          fw_increases = 0
        else:
          fw_increases += 1

        
        if loss_bw_val < lowest_loss_bw_val:
          lowest_loss_bw_val = loss_bw_val
          bw_increases = 0
        else:
          bw_increases += 1
        
        print("fw_increases:",fw_increases,"bw_increases:",bw_increases)
        
        if fw_increases >= 10 and bw_increases >10:
          print("Early stopping triggered")
          break  # Exit the training loop

    # Plot losses of the fw task
    plt.figure()
    plt.plot(epochs_list, loss_fw_train_list, label='Training')
    plt.plot(epochs_list, loss_fw_val_list, label='Validation')
    plt.plot(epochs_list, loss_fw_test_list, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses of the FW Task')
    plt.legend()
    plt.savefig('plot1.png')

    # Plot losses of the bw task
    plt.figure()
    plt.plot(epochs_list, loss_bw_train_list, label='Training')
    plt.plot(epochs_list, loss_bw_val_list, label='Validation')
    plt.plot(epochs_list, loss_bw_test_list, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses of the BW Task')
    plt.legend()
    plt.savefig('plot2.png')

    # Plot accuracies
    plt.figure()
    plt.plot(epochs_list, acc_train_list, label='Training')
    plt.plot(epochs_list, acc_val_list, label='Validation')
    plt.plot(epochs_list, acc_test_list, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracies')
    plt.legend()
    plt.savefig('plot3.png')

def eval_model(model, test_loader, get_x_y, loss_fw, loss_bw, b1_stochastic, b2_stochastic, b3_stochastic, cuda):
    correct = 0
    total = len(test_loader.dataset)
    loss_fw_train = 0
    loss_bw_train = 0
    model.eval()    
    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):

            x, y = get_x_y(batch, cuda)
            if torch.cuda.is_available(): 
              x = x.to("cuda")
              y = y.to("cuda")
            outputs, _, _ = model.forward(x, b1_stochastic, b2_stochastic, b3_stochastic)
            predicted = outputs.argmax(dim=1)
            # print(predicted.device)
            # print(y.device)
            correct += (predicted == y).sum().item()

            logits_bw, _ = model.backward(y, b1_stochastic, b2_stochastic)
            loss_val_bw = loss_bw(logits_bw, (x>0).to(dtype=x.dtype))
            loss_bw_train += loss_val_bw.mean(dim=-1).sum().item()

            loss_val_fw = loss_fw(outputs, y)
            loss_avg = loss_val_fw.mean()
            loss_fw_train += loss_val_fw.sum().item()

    accuracy = correct/total
    fw_loss = loss_fw_train/total
    bw_loss = loss_bw_train/total
    return accuracy, fw_loss, bw_loss

if __name__ == '__main__':
    main()