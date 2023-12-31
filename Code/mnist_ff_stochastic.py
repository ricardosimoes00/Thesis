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
  shannon_entropy = shannon_entropy.view(1, -1)
  
  return - shannon_entropy

def reg_relu(k):

  k_copy = k.clone()
  k_flat = k_copy.view(k_copy.size(0), -1)
  euclidean_norm = torch.norm(k_flat, dim=1)
  euclidean_norm = euclidean_norm.view(1, -1)
  
  return 0.5 * torch.square(euclidean_norm)

def reg_tanh(k):
  result_tensor = (k + 1) / 2 * torch.log((k + 1) / 2) + (k - 1) / 2 * torch.log((k - 1) / 2)
  sum_tensor = torch.sum(result_tensor, dim=1)  # shape: [batch_size]
  return sum_tensor


def load_config(conf, show=False):
    # conf = OmegaConf.from_cli()

    # validate against schema
    schema = OmegaConf.structured(MNISTConvConfigSchema)
    conf = OmegaConf.merge(schema, conf)

    if show:
        print(OmegaConf.to_yaml(conf))

    conf = OmegaConf.to_container(conf)

    return conf

def get_x_y(batch, cuda):
    # *_, w, h = batch[0].shape
    # images = batch[0].view(-1, w * h)  # between [0, 1].
    images = batch[0]
    images = 2*images - 1  # [between -1 and 1]
    labels = batch[1]

    if cuda:
        images = images.cuda()
        labels = labels.cuda()

    return images, labels
    
# @dataclass
# class MNISTConvConfigSchema:
#     dropout: float = 0.3
#     seed: int = 42
#     lr: float = 0.01
#     batch_size: int = 512
#     max_epochs: int = 100
#     unn_iter: int = 1
#     unn_y_init: str = "zero"
#     backward_loss_coef: float = 0.1 #REFAZER COM ESTE VALOR
#     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
#     noise_sd: float = 0 
#     noise_mean: float = 0
#     activation: str = "relu"
    

class FF_UNN(torch.nn.Module):

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
        n1 = 784
        n2 = 128
        n3 = n_classes

        self.d0 = d0
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
        self.W1 = torch.nn.Parameter(torch.empty(n1, n2))
        self.b1 = torch.nn.Parameter(torch.empty(n2))

        self.W2 = torch.nn.Parameter(torch.empty(n2, n3))
        self.b2 = torch.nn.Parameter(torch.empty(n3))

        print("self.W1", self.W1.shape)
        print("self.b1", self.b1.shape)
        print("self.W2", self.W2.shape)
        print("self.b2", self.b2.shape)

        torch.manual_seed(self.seed)

        torch.nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
                                      # nonlinearity='tanh')
        torch.nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
                                      # nonlinearity='tanh')

        fan_in_1, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W1)
        bound_1 = 1 / math.sqrt(fan_in_1)
        torch.nn.init.uniform_(self.b1, -bound_1, bound_1)

        fan_in_2, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W2)
        bound_2 = 1 / math.sqrt(fan_in_2)
        torch.nn.init.uniform_(self.b2, -bound_2, bound_2)        

    
    def _update_X(self, H1):

        X_updated = torch.matmul(H1, self.W1.transpose(0, 1))

        return X_updated

    def _update_H1(self, X, Y, b1_stochastic):

        
        H1_fwd = torch.matmul(X, self.W1) + (self.b1 + b1_stochastic)
        H1_bwd = torch.matmul(Y, self.W2.transpose(0, 1))
        
        H1_updated = H1_fwd + H1_bwd

        return H1_updated

    def _update_Y(self, H1, b2_stochastic):
        
        Y_updated = torch.matmul(H1, self.W2) + (self.b2 + b2_stochastic)

        return Y_updated

    def forward(self, X, b1_stochastic, b2_stochastic):

        # self.activation = "relu"

        if self.y_init == "zero":
            Y = torch.zeros(1, self.n3, device=X.device) 
        elif self.y_init == "rand":
            Y = torch.rand(1, self.n3, device=X.device) 
            Y = torch.softmax(Y, dim=-1)
        elif self.y_init == "uniform":
            Y = torch.zeros(1, self.n3, device=X.device) 
            Y = torch.softmax(Y, dim=-1)

        b = X.shape[0]
        mask_H1 = torch.ones(b, self.n2, device=X.device)
        mask_H1 = self.dropout(mask_H1)
      
        total_energy = []
        total_entropy = []
        
        for i in range(self.k):
            if i==0: # Standard training of UNN with k steps of coordinate descent
                
                X_flattened = torch.flatten(X, 1, 3)
                
                if torch.cuda.is_available(): 
                  device = torch.device("cuda")
                  X_flattened = X_flattened.to(device)
                  Y = Y.to(device)
                  b1_stochastic = b1_stochastic.to(device)
                else:
                  device = torch.device("cpu")
                  X_flattened = X_flattened.to(device)
                  Y = Y.to(device)
                  b1_stochastic = b1_stochastic.to(device)

                H1 = self._update_H1(X_flattened, Y, b1_stochastic)
                
                if self.activation == "relu":
                  H1 = torch.relu(H1)
                elif self.activation == "tanh":
                  H1 = torch.tanh(H1)

                if torch.cuda.is_available(): 
                  device = torch.device("cuda")
                  mask_H1 = mask_H1.to(device)
                  H1 = H1.to(device)
                else:
                    device = torch.device("cpu")
                    mask_H1 = mask_H1.to(device)
                    H1 = H1.to(device)

                H1 = H1 * mask_H1

                if torch.cuda.is_available(): 
                  device = torch.device("cuda")
                  b2_stochastic = b2_stochastic.to(device)
                else:
                  device = torch.device("cpu")
                  b2_stochastic = b2_stochastic.to(device)
              

                Y_logits = self._update_Y(H1, b2_stochastic)
                Y = torch.softmax(Y_logits, dim=-1)

                entropy = reg_softmax(Y)
                total_entropy.append(-torch.sum(entropy))

                # print("X.device:",X.device)
                # print("H1.device:",H1.device)
                # print("Y.device:",Y.device)
                # print("b1_stochastic.device:",b1_stochastic.device)
                # print("b2_stochastic.device:",b2_stochastic.device)
                if torch.cuda.is_available():
                  X = X.to("cuda")
                  
                # print("reg_softmax(Y).t().squeeze(1).shape:",reg_softmax(Y).t().squeeze(1).shape)
                # print("reg_relu(H1).t().squeeze(1).shape:",reg_relu(H1).t().squeeze(1).shape)
                # print("reg_tanh(H1).shape:",reg_tanh(H1).shape)

                if self.activation == "relu":
                  energy_y = - torch.matmul(Y, (self.b2 + b2_stochastic)) + reg_softmax(Y).t().squeeze(1) - torch.sum(Y*torch.matmul(H1, self.W2),dim=1)
                  energy_h1 = - torch.matmul(H1, (self.b1 + b1_stochastic)) + reg_relu(H1).t().squeeze(1) - torch.sum(H1*torch.matmul(torch.flatten(X, 1, 3), self.W1),dim=1)
           
                elif self.activation == "tanh":
                  energy_y = - torch.matmul(Y, (self.b2 + b2_stochastic)) + reg_softmax(Y).t().squeeze(1) - torch.sum(Y*torch.matmul(H1, self.W2),dim=1)
                  energy_h1 = - torch.matmul(H1, (self.b1 + b1_stochastic)) + reg_tanh(H1) - torch.sum(H1*torch.matmul(torch.flatten(X, 1, 3), self.W1),dim=1)

                total_energy.append(energy_y + energy_h1)
        
        return Y_logits, total_energy, total_entropy

    def backward(self, y, b1_stochastic, k=None, return_all_x=True):

        if not k:
            k = self.k

        Y = torch.nn.functional.one_hot(y, num_classes=self.n_classes)

        b = Y.shape[0]

        all_X = []

        H1 = torch.zeros(b, self.n1, self.n2, device=y.device)

        X = torch.zeros(b, 1, self.d0, self.d0, device=y.device)
        X = torch.flatten(X, 1, 3)
        Y = Y.to(dtype=X.dtype)

        for i in range(self.k):
            H1 = self._update_H1(X, Y, b1_stochastic)
            
            if self.activation == "relu":
              H1 = torch.relu(H1)
            elif self.activation == "tanh":
              H1 = torch.tanh(H1)

            Xp = self._update_X(H1)
            Xp = torch.reshape(Xp, (b,1,28,28))

            if self.activation == "relu":
              X = torch.relu(Xp)
            elif self.activation == "tanh":
              X = torch.tanh(Xp)
            # X_reshaped = X.reshape(512, 1, 28, 28)

            if return_all_x:
                all_X.append(X.detach().clone())

            if self.k > 1:
              X = torch.flatten(X, 1, 3)
                  
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

        model = FF_UNN(k=conf['unn_iter'],
                        n_classes=output_dim,
                        dropout_p=conf['dropout'],
                        y_init=conf['unn_y_init'],
                        seed = conf["seed"],
                        activation = conf["activation"])

        if torch.cuda.is_available(): 
          model = model.cuda()

        print(model)

        # optimizer = torch.optim.SGD(model.parameters(), lr=conf['lr'], momentum = 0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])

        print('Num parameters:', sum(p.numel() for p in model.parameters()))
      
        
        for name, param in model.named_parameters():
            print(name, param.numel())

        #torch.manual_seed(conf['seed'])

        #b1_stochastic = torch.randn(b1_shape)*conf["noise_sd"] + conf["noise_mean"]
        #b2_stochastic = torch.randn(b2_shape)*conf["noise_sd"] + conf["noise_mean"]

        noise_sd = conf["noise_sd"]
        noise_mean = conf["noise_mean"]

        train_model(model,
                    train_loader,
                    dev_loader,
                    test_loader,
                    optimizer,
                    conf,
                    partial(get_x_y, cuda=conf['device'] == 'cuda'),
                    noise_sd,
                    noise_mean
                    )
                    
        run.finish()



def train_model(model, train_loader, val_loader, test_loader,
                optimizer, conf, get_x_y, noise_sd, noise_mean):

    n_train = len(train_loader.dataset)

    best_val_acc = 0
    best_val_acc_test_acc = None
    best_val_acc_epoch = None

    # accuracy_val = eval_model(model, val_loader, get_x_y)
    # accuracy_test = eval_model(model, test_loader, get_x_y)
    # print("Before training acc", accuracy_val, accuracy_test)

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

    W1_shape, b1_shape, W2_shape, b2_shape = params_shape_list
    
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
        
        noise_seeds_list.append(random_int)
       
        loss_fw_train = 0
        loss_bw_train = 0
        accuracy_train = 0

        loss_fw_val_total = 0 
        loss_bw_val_total = 0 
        
        for batch_id, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x, y = get_x_y(batch)

            # [batch x n_classes]
            if torch.cuda.is_available:
              x = x.to("cuda")
              y = y.to("cuda")
              b1_stochastic = b1_stochastic.to("cuda")
            logits_fw, _, _ = model(x, b1_stochastic, b2_stochastic)
            # [batch]
            
            loss_val_fw = loss_fw(logits_fw, y)

            loss_avg = loss_val_fw.mean()
            if conf['backward_loss_coef'] > 0:
                
                # [batch x 1 x 28 x 28]

                logits_bw, _ = model.backward(y, b1_stochastic)

                # [batch x 1 x 28 x 28]
                # print(b1_stochastic.device)
                # print(y.device)

                loss_val_bw = loss_bw(logits_bw, (x>0).to(dtype=x.dtype))

                loss_avg = loss_avg + conf['backward_loss_coef'] * loss_val_bw.mean()

            
            loss_avg.backward()
            optimizer.step()
            loss_fw_train += loss_val_fw.sum().item()
            if conf['backward_loss_coef'] > 0:
                loss_bw_train += loss_val_bw.mean(dim=-1).sum().item()
            accuracy_train += (logits_fw.argmax(dim=1) == y).sum().item()

        accuracy_val, loss_fw_val, loss_bw_val = eval_model(model, val_loader, get_x_y, loss_fw, loss_bw, b1_stochastic, b2_stochastic)
        accuracy_test, loss_fw_test, loss_bw_test = eval_model(model, test_loader, get_x_y, loss_fw, loss_bw, b1_stochastic, b2_stochastic)

        loss_val_avg = loss_fw_val + conf["backward_loss_coef"]*loss_bw_val

        loss_fw_train /= n_train  # average sample loss
        loss_bw_train /= n_train  # average sample loss
        
        # loss_fw_val /= n_train
        # loss_bw_val /= n_train
        # loss_fw_test /= n_train
        # loss_bw_test /= n_train

        accuracy_train /= n_train

        if accuracy_val > best_val_acc:
            best_val_acc = accuracy_val
            best_val_acc_test_acc = accuracy_test
            best_val_acc_epoch = epoch
            
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
        acc_test_list .append(accuracy_test)
        
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

    
def eval_model(model, test_loader, get_x_y, loss_fw, loss_bw, b1_stochastic, b2_stochastic):
    correct = 0
    total = len(test_loader.dataset)
    loss_fw_train = 0
    loss_bw_train = 0
    model.eval()    
    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):

            x, y = get_x_y(batch)
            if torch.cuda.is_available(): 
              x = x.to("cuda")
              y = y.to("cuda")
            outputs, _, _ = model.forward(x, b1_stochastic, b2_stochastic)
            predicted = outputs.argmax(dim=1)
            # print(predicted.device)
            # print(y.device)
            correct += (predicted == y).sum().item()

            logits_bw, _ = model.backward(y, b1_stochastic)
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