import math
import numpy as np
from functools import partial
from dataclasses import dataclass
from typing import Optional
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
import argparse
import ast
import sys
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score

def variance(tensor_list_logits):

  tensor_logits = torch.stack(tensor_list_logits)
  model_probabilities = torch.softmax(tensor_logits, dim = 2)
  average_probabilities = torch.mean(model_probabilities, dim=0)
  variance_probs_per_class = torch.var(model_probabilities, dim=0)
  variance_across_images = torch.mean(variance_probs_per_class, dim=1)

  return variance_across_images

def mean_accuracy(tensor_list_logits, test_labels):
  
  tensor_logits = torch.stack(tensor_list_logits)
  model_probabilities = torch.softmax(tensor_logits, dim = 2)
  average_probabilities = torch.mean(model_probabilities, dim = 0)
  model_predictions = torch.argmax(average_probabilities, dim = 1)
  
  return torch.sum( model_predictions == test_labels)/len(test_labels) * 100

def entropy_metric(tensor_list_logits):
  
  tensor_logits = torch.stack(tensor_list_logits)
  model_probabilities = torch.softmax(tensor_logits, dim = 2)
  average_probailities = torch.mean(model_probabilities, dim = 0) 
  average_probailities[average_probailities == 0] = 1
  transformed_tensor = torch.log(average_probailities)*average_probailities 
  entropy_sum = torch.sum(transformed_tensor, dim=1)

  return -entropy_sum

def jsd_metric(tensor_list_logits): 

  tensor_logits = torch.stack(tensor_list_logits)
  model_probabilities = torch.softmax(tensor_logits, dim = 2)
  model_probabilities[model_probabilities == 0] = 1
  num_iterations, num_images, num_classes = model_probabilities.shape
  jsd_results = torch.zeros((num_images,))

  for i in range(num_images):

      p = model_probabilities[:, i, :] #[50, 10]
      q = torch.mean(model_probabilities[:, i, :], dim=0) #[10]
      kl = (p * (p.log() - q.log())).sum(dim=1)
      average_jsd = kl.mean(dim = 0)
      jsd_results[i] = average_jsd

  return jsd_results

def max_probability_metric(tensor_list_logits):

  tensor_logits = torch.stack(tensor_list_logits)
  model_probabilities = torch.softmax(tensor_logits, dim = 2)
  average_probs = torch.mean(model_probabilities, dim = 0)
  max_p_values, _ = torch.max(average_probs, dim=1)
  
  return max_p_values

def mi_metric(tensor_list_logits):

  expected_entropy = entropy_metric(tensor_list_logits)

  prob = torch.nn.functional.softmax(torch.stack(tensor_list_logits), dim = 2) #[50, 10000, 10]
  mean_prob = torch.mean(prob, dim  = 0) #[10000, 10]
  mean_prob[mean_prob == 0] = 1 #[10000, 10]
  transformed_tensor = torch.log(mean_prob)*mean_prob #[10000, 10]
  entropy_sum = -torch.sum(transformed_tensor, dim=1)

  return entropy_sum - expected_entropy

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(tensor_list_logits, tensor_list_noisy_logits_25, tensor_list_noisy_logits_50, tensor_list_noisy_logits_75, test_labels):

    from sklearn.metrics import precision_recall_curve, auc
    set_seed(42)

    entropy = entropy_metric(tensor_list_logits)
    noisy_entropy_50 = entropy_metric(tensor_list_noisy_logits_50)
    
    jsd = jsd_metric(tensor_list_logits)
    noisy_jsd_50 = jsd_metric(tensor_list_noisy_logits_50)
    
    acc = mean_accuracy(tensor_list_logits, test_labels)
    noisy_acc_50 = mean_accuracy(tensor_list_noisy_logits_50, test_labels)
 
    comparisson = (torch.argmax(torch.softmax(torch.mean(torch.stack(tensor_list_logits), dim = 0), dim = 1), dim = 1) == test_labels)
    noisy_comparisson = (torch.argmax(torch.softmax(torch.mean(torch.stack(tensor_list_noisy_logits_50), dim = 0), dim = 1), dim = 1) == test_labels)

    def calculate_auc(labels, uncertainty_scores):
        fpr, tpr, _ = roc_curve(labels, uncertainty_scores)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def calculate_aupr(labels, uncertainty_scores):
        precision, recall, _ = precision_recall_curve(labels, uncertainty_scores)
        aupr = auc(recall, precision)
        return aupr


    mi_scores = mi_metric(tensor_list_logits)
    max_prob_scores = max_probability_metric(tensor_list_logits)
    entropy_scores = entropy_metric(tensor_list_logits)
    mi_auc = calculate_auc(comparisson.cpu(), mi_scores.cpu())
    max_prob_auc = calculate_auc(comparisson.cpu(), max_prob_scores.cpu())
    entropy_auc = calculate_auc(comparisson.cpu(), entropy_scores.cpu())
    mi_aupr = calculate_aupr(comparisson.cpu(), mi_scores.cpu())
    max_prob_aupr = calculate_aupr(comparisson.cpu(), max_prob_scores.cpu())
    entropy_aupr = calculate_aupr(comparisson.cpu(), entropy_scores.cpu())

    noisy_mi_scores = mi_metric(tensor_list_noisy_logits_50)
    noisy_max_prob_scores = max_probability_metric(tensor_list_noisy_logits_50)
    noisy_entropy_scores = entropy_metric(tensor_list_noisy_logits_50)
    noisy_mi_auc = calculate_auc(noisy_comparisson.cpu(), noisy_mi_scores.cpu())
    noisy_max_prob_auc = calculate_auc(noisy_comparisson.cpu(), noisy_max_prob_scores.cpu())
    noisy_entropy_auc = calculate_auc(noisy_comparisson.cpu(), noisy_entropy_scores.cpu())
    noisy_mi_aupr = calculate_aupr(noisy_comparisson.cpu(), noisy_mi_scores.cpu())
    noisy_max_prob_aupr = calculate_aupr(noisy_comparisson.cpu(), noisy_max_prob_scores.cpu())
    noisy_entropy_aupr = calculate_aupr(noisy_comparisson.cpu(), noisy_entropy_scores.cpu())


    print("Mean Acc:", acc.item())
    print("Mean Entropy:", torch.mean(entropy).item())
    print("Median Entropy:", torch.median(entropy).item())
    print("Mean JSD:", torch.mean(jsd).item())
    print("Median JSD:", torch.median(jsd).item())
    print("Mean Variance:", torch.mean(variance(tensor_list_logits)).item())
    print("Median Variance:", torch.median(variance(tensor_list_logits)).item())
    print(f'MI AUROC: {mi_auc:.4f}')
    print(f'Max Probability AUROC: {max_prob_auc:.4f}')
    print(f'Entropy AUROC: {entropy_auc:.4f}')
    print(f'MI AUPR: {mi_aupr:.4f}')
    print(f'Max Probability AUPR: {max_prob_aupr:.4f}')
    print(f'Entropy AUPR: {entropy_aupr:.4f}')
    print()
    # print(torch.mean(entropy).shape)
    # print(torch.mean(entropy).shape)

    # print("Mean Acc:", noisy_acc_25.item())
    # print("Mean Entropy:", torch.mean(noisy_entropy_25).item())
    # print("Median Entropy:", torch.median(noisy_entropy_25).item())
    # print("Mean JSD:", torch.mean(noisy_jsd_25).item())
    # print("Median JSD:", torch.median(noisy_jsd_25).item())
    # print("Mean Variance:", torch.mean(variance(tensor_list_noisy_logits_25)).item())
    # print("Median Variance:", torch.median(variance(tensor_list_noisy_logits_25)).item())
    # print()

    print("Mean Acc:", noisy_acc_50.item())
    print("Mean Entropy:", torch.mean(noisy_entropy_50).item())
    print("Median Entropy:", torch.median(noisy_entropy_50).item())
    print("Mean JSD:", torch.mean(noisy_jsd_50).item())
    print("Median JSD:", torch.median(noisy_jsd_50).item())
    print("Mean Variance:", torch.mean(variance(tensor_list_noisy_logits_50)).item())
    print("Median Variance:", torch.median(variance(tensor_list_noisy_logits_50)).item())
    print(f'MI AUROC: {noisy_mi_auc:.4f}')
    print(f'Max Probability AUROC: {noisy_max_prob_auc:.4f}')
    print(f'Entropy AUROC: {noisy_entropy_auc:.4f}')
    print(f'MI AUPR: {noisy_mi_aupr:.4f}')
    print(f'Max Probability AUPR: {noisy_max_prob_aupr:.4f}')
    print(f'Entropy AUPR: {noisy_entropy_aupr:.4f}')
    print()

    # print("Mean Acc:", noisy_acc_75.item())
    # print("Mean Entropy:", torch.mean(noisy_entropy_75).item())
    # print("Median Entropy:", torch.median(noisy_entropy_75).item())
    # print("Mean JSD:", torch.mean(noisy_jsd_75).item())
    # print("Median JSD:", torch.median(noisy_jsd_75).item())
    # print("Mean Variance:", torch.mean(variance(tensor_list_noisy_logits_75)).item())
    # print("Median Variance:", torch.median(variance(tensor_list_noisy_logits_75)).item())
    # print()

    # print("Mean noisy-Acc (sd = 0.25):", noisy_acc_25.item())
    # print("Mean noisy-Acc (sd = 0.50):", noisy_acc_50.item())
    # print("Mean noisy-Acc (sd = 0.75):", noisy_acc_75.item())

    # simplex_plot(tensor_list_logits)
    # simplex_plot(tensor_list_noisy_logits_25)
    # simplex_plot(tensor_list_noisy_logits_50)
    # simplex_plot(tensor_list_noisy_logits_75)

if __name__ == '__main__':

    tensor_list_logits_path = sys.argv[1]
    tensor_list_noisy_logits_25_path = sys.argv[2]
    tensor_list_noisy_logits_50_path = sys.argv[3]
    tensor_list_noisy_logits_75_path = sys.argv[4]
    test_labels_path = sys.argv[5]
    

    tensor_list_logits = torch.load(tensor_list_logits_path)
    test_labels = torch.load(test_labels_path)
    tensor_list_noisy_logits_25 = torch.load(tensor_list_noisy_logits_25_path)
    tensor_list_noisy_logits_50 = torch.load(tensor_list_noisy_logits_50_path)
    tensor_list_noisy_logits_75 = torch.load(tensor_list_noisy_logits_75_path)


    main(tensor_list_logits, tensor_list_noisy_logits_25, tensor_list_noisy_logits_50, tensor_list_noisy_logits_75, test_labels)	