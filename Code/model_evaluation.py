import math
import numpy as np
from functools import partial
from dataclasses import dataclass
from typing import Optional
from omegaconf import OmegaConf
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
import argparse
import ast
import sys
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix

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
  
  return torch.sum( model_predictions == (test_labels))/len(test_labels) * 100

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

def entropy_metric(tensor_list_logits):
  
  tensor_logits = torch.stack(tensor_list_logits)
  model_probabilities = torch.softmax(tensor_logits, dim = 2)
  average_probailities = torch.mean(model_probabilities, dim = 0) 
  average_probailities[average_probailities == 0] = 1
  transformed_tensor = torch.log(average_probailities)*average_probailities 
  entropy_sum = torch.sum(transformed_tensor, dim=1)

  return -entropy_sum

def mi_metric(tensor_list_logits):

  expected_of_average = entropy_metric(tensor_list_logits)
  tensor_logits = torch.stack(tensor_list_logits)
  prob = torch.softmax(tensor_logits, dim = 2) #[50, 10000, 10]
  prob[prob == 0] = 1
  transformed_tensor = torch.log(prob)*prob #[10000, 10]
  entropy = -torch.sum(transformed_tensor, dim=2)
  average_entropy = torch.mean(entropy, dim  = 0)

  return expected_of_average - average_entropy

def set_seed(seed):
   
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(tensor_list_logits, tensor_list_noisy_logits_50, test_labels):

    set_seed(42)
 
    entropy = entropy_metric(tensor_list_logits)
    noisy_entropy_50 = entropy_metric(tensor_list_noisy_logits_50)
 
    jsd = jsd_metric(tensor_list_logits)
    noisy_jsd_50 = jsd_metric(tensor_list_noisy_logits_50)
 
    acc = mean_accuracy(tensor_list_logits, test_labels)
    noisy_acc_50 = mean_accuracy(tensor_list_noisy_logits_50, test_labels)

    #OLD
    # comparisson = (torch.argmax(torch.softmax(torch.mean(torch.stack(tensor_list_logits), dim = 0), dim = 1), dim = 1) != test_labels)
    # noisy_comparisson = (torch.argmax(torch.softmax(torch.mean(torch.stack(tensor_list_noisy_logits_50), dim = 0), dim = 1), dim = 1) != test_labels)

    #New   
    comparisson = (torch.argmax(torch.mean(torch.softmax(torch.stack(tensor_list_logits), dim = 2), dim = 0), dim = 1) != test_labels)
    noisy_comparisson = (torch.argmax(torch.mean(torch.softmax(torch.stack(tensor_list_noisy_logits_50), dim = 2), dim = 0), dim = 1) != test_labels)

    def calculate_auroc(labels, uncertainty_scores):
      
        fpr, tpr, _ = roc_curve(labels, uncertainty_scores)
        roc_auc = auc(fpr, tpr)
      
        return roc_auc

    def calculate_aupr(labels, uncertainty_scores):
      
        precision, recall, _ = precision_recall_curve(labels, uncertainty_scores)
        aupr = auc(recall, precision)
      
        return aupr

    #Missclassification experiment 
    max_prob_scores = max_probability_metric(tensor_list_logits)
    entropy_scores = entropy_metric(tensor_list_logits)
    mi_scores = mi_metric(tensor_list_logits)
    jsd_scores = jsd_metric(tensor_list_logits)
    
    max_prob_auc = calculate_auroc(comparisson.cpu(), max_prob_scores.cpu())
    entropy_auc = calculate_auroc(comparisson.cpu(), entropy_scores.cpu())
    mi_auc = calculate_auroc(comparisson.cpu(), mi_scores.cpu())
    jsd_auc = calculate_auroc(comparisson.cpu(), jsd_scores.cpu())
    
    max_prob_aupr = calculate_aupr(comparisson.cpu(), max_prob_scores.cpu())
    entropy_aupr = calculate_aupr(comparisson.cpu(), entropy_scores.cpu())
    mi_aupr = calculate_aupr(comparisson.cpu(), mi_scores.cpu())
    jsd_aupr = calculate_aupr(comparisson.cpu(), jsd_scores.cpu())
       
    tensor_list_noisy_logits_50_merged = torch.cat((torch.stack(tensor_list_noisy_logits_50),torch.stack(tensor_list_logits)), dim = 1)
    num_ones = torch.stack(tensor_list_noisy_logits_50).shape[1]
    num_zeros = torch.stack(tensor_list_logits).shape[1]
    ones_tensor = torch.ones(num_ones)
    zeros_tensor = torch.zeros(num_zeros)
    noisy_comparisson = torch.cat((ones_tensor, zeros_tensor), dim=0)
    
    # print(num_ones)
    # print(num_zeros)
    # print(noisy_comparisson.shape)

    # print(len(tensor_list_noisy_logits_50_merged))

    # Create the plot1
    plt.figure(figsize=(8, 6))
    plt.scatter(max_prob_scores.cpu().numpy(), jsd_scores.cpu().numpy(), s=10)  # s controls the marker size

    # Add labels and a title
    plt.xlabel('max_probability_metric')
    plt.ylabel('jsd_metric')
    plt.title('Comparison of max_probability_metric and jsd_metric')

    x_interval = (0.95, 1)  # Define your desired x-axis interval
    y_interval = (0,0.02)  # Define your desired y-axis interval
    plt.xlim(x_interval)
    plt.ylim(y_interval)


    plt.savefig('comparison_plot1.png')

    # Create the plot2
    plt.figure(figsize=(8, 6))
    plt.scatter(max_prob_scores.cpu().numpy(), entropy_scores.cpu().numpy(), s=10)  # s controls the marker size

    # Add labels and a title
    plt.xlabel('max_probability_metric')
    plt.ylabel('entropy_scores')
    plt.title('Comparison of max_probability_metric and entropy_scores')
   
    x_interval = (0.95, 1)  # Define your desired x-axis interval
    y_interval = (0,0.02)  # Define your desired y-axis interval
    plt.xlim(x_interval)
    plt.ylim(y_interval)
   
    plt.savefig('comparison_plot2.png')

    # Create the plot3
    plt.figure(figsize=(8, 6))
    plt.scatter(max_prob_scores.cpu().numpy(), mi_scores.cpu().numpy(), s=10)  # s controls the marker size

    # Add labels and a title
    plt.xlabel('max_probability_metric')
    plt.ylabel('mi_scores')
    plt.title('Comparison of max_probability_metric and mi_scores')

    x_interval = (0.95, 1)  # Define your desired x-axis interval
    y_interval = (0,0.02)  # Define your desired y-axis interval
    plt.xlim(x_interval)
    plt.ylim(y_interval)

    plt.savefig('comparison_plot3.png')

    #OOD experiment   
    # print("tensor_list_noisy_logits_50_merged.shape",torch.stack(tensor_list_noisy_logits_50_merged).shape)
    # print("tensor_list_logits.shape",torch.stack(tensor_list_logits).shape)
    tensor_list_noisy_logits_50_merged = torch.unbind(tensor_list_noisy_logits_50_merged, dim = 0)
    noisy_max_prob_scores = max_probability_metric(tensor_list_noisy_logits_50_merged)
    noisy_entropy_scores = entropy_metric(tensor_list_noisy_logits_50_merged)
    noisy_mi_scores = mi_metric(tensor_list_noisy_logits_50_merged)
    noisy_jsd_scores = jsd_metric(tensor_list_noisy_logits_50_merged)
    
    # print("comparisson.shape",comparisson.shape)
    # print("max_prob_scores.shape",max_prob_scores.shape)
    # print("noisy_comparisson.shape",noisy_comparisson.shape)
    # print("noisy_max_prob_scores.shape:",noisy_max_prob_scores.shape)
    
    noisy_max_prob_auc = calculate_auroc(noisy_comparisson.cpu(), noisy_max_prob_scores.cpu())
    noisy_entropy_auc = calculate_auroc(noisy_comparisson.cpu(), noisy_entropy_scores.cpu())
    noisy_mi_auc = calculate_auroc(noisy_comparisson.cpu(), noisy_mi_scores.cpu())
    noisy_jsd_auc = calculate_auroc(noisy_comparisson.cpu(), noisy_jsd_scores.cpu())

    noisy_max_prob_aupr = calculate_aupr(noisy_comparisson.cpu(), noisy_max_prob_scores.cpu())
    noisy_entropy_aupr = calculate_aupr(noisy_comparisson.cpu(), noisy_entropy_scores.cpu())
    noisy_mi_aupr = calculate_aupr(noisy_comparisson.cpu(), noisy_mi_scores.cpu())
    noisy_jsd_aupr = calculate_aupr(noisy_comparisson.cpu(), noisy_jsd_scores.cpu())

    print("Missclassification detection experiment")
    print("Mean Acc:", acc.item())
    
    print(f'Max Probability AUROC: {max_prob_auc:.4f}')
    print(f'Entropy AUROC: {entropy_auc:.4f}')
    print(f'MI AUROC: {mi_auc:.4f}')
    print(f'JSD AUROC: {jsd_auc:.4f}')
    
    print(f'Max Probability AUPR: {max_prob_aupr:.4f}')
    print(f'Entropy AUPR: {entropy_aupr:.4f}')
    print(f'MI AUPR: {mi_aupr:.4f}')
    print(f'JSD AUPR: {jsd_aupr:.4f}')
    print()
    
    print("OOD detection experiment")
    print(f'Max Probability AUROC: {noisy_max_prob_auc:.4f}')
    print(f'Entropy AUROC: {noisy_entropy_auc:.4f}')
    print(f'MI AUROC: {noisy_mi_auc:.4f}')
    print(f'JSD AUROC: {noisy_jsd_auc:.4f}')
    print(f'Max Probability AUPR: {noisy_max_prob_aupr:.4f}')
    print(f'Entropy AUPR: {noisy_entropy_aupr:.4f}')
    print(f'MI AUPR: {noisy_mi_aupr:.4f}')
    print(f'JSD AUPRR: {noisy_jsd_aupr:.4f}')
    print()

    #missclass or OOD, metrica, pr ou roc

    # Prompt the user for the experiment choice
    # Prompt the user for the experiment choice
    experiment_choice = (input("Enter 1 for Misclassification Experiment or 2 for OOD Detection Experiment: "))
    threshold = float(input("Enter threshold value: "))
    print("threshold value selected:", threshold)

    if experiment_choice == "1" or experiment_choice == "2":
        print("ola1")

        curve_type_choice = (input("Enter 1 for PR Curve or 2 for ROC Curve: "))

        if curve_type_choice == '1' or curve_type_choice == '2':
            print("ola2")

            metric_choice = (input("Enter 1 for Entropy, 2 for MI, 3 for JSD, or 4 for Max Prob: "))

            if metric_choice in ('1', '2', '3', '4'):
                if experiment_choice == "1":  #missclass
                    if curve_type_choice == '1': #PR curve
                        print("ola2.5")
                        if metric_choice == '1':
                          metric = entropy_scores.cpu()
                          precision, recall, thresholds = precision_recall_curve(comparisson.cpu(), metric)

                        elif metric_choice == '2':
                          metric = mi_scores.cpu()
                          precision, recall, thresholds = precision_recall_curve(comparisson.cpu(), metric)

                        elif metric_choice == '3':
                          metric = jsd_scores.cpu()
                          precision, recall, thresholds = precision_recall_curve(comparisson.cpu(), metric)

                        elif metric_choice == '4':
                          metric = max_prob_scores.cpu()
                          precision, recall, thresholds = precision_recall_curve(comparisson.cpu(), metric)
                        print("ola3")
                
                        # Calculate the AUPR
                        aupr = auc(recall, precision)

                        # Plot the PR curve
                        plt.figure(figsize=(8, 6))
                        plt.plot(recall, precision, lw=2)
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title('Precision-Recall Curve for FFNN-type SUNN')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig('precision_recall_curve.png')
                        plt.show()

                        # Print the AUPR
                        print("AUPR:", aupr)
                        
                        binary_predictions = [1 if p >= threshold else 0 for p in metric]
                        confusion = confusion_matrix(comparisson.cpu(), binary_predictions)
                        print(confusion)
                      
                        cmap = plt.matplotlib.colors.ListedColormap(['grey', 'grey', 'grey', 'grey'])
                        class_labels = ["Negative", "Positive"]  # Horizontal labels

                        # Create a confusion matrix plot without the color scale
                        plt.figure(figsize=(6, 4))
                        plt.imshow(confusion, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
                        plt.title(f'Confusion Matrix for Threshold={threshold}')
                        tick_marks = np.arange(len(class_labels))
                        plt.xticks(tick_marks, class_labels)  # Horizontal class labels
                        plt.yticks(tick_marks, class_labels)  # Horizontal class labels
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        for i in range(len(class_labels)):
                            for j in range(len(class_labels)):
                                text_color = "black" if confusion[i, j] == 0 else "white"
                                plt.text(j, i, format(confusion[i, j], 'd'), horizontalalignment="center", color=text_color)
                        plt.savefig('cm.png')
                        plt.show()


                    elif curve_type_choice == '2': #Roc curve

                        if metric_choice == '1':
                          metric = entropy_scores.cpu()
                          fpr, tpr, thresholds = roc_curve(comparisson.cpu(), metric)

                        elif metric_choice == '2':
                          metric = mi_scores.cpu()
                          fpr, tpr, thresholds =   roc_curve(comparisson.cpu(), metric)

                        elif metric_choice == '3':
                          metric = jsd_scores.cpu()
                          fpr, tpr, thresholds = roc_curve(comparisson.cpu(), metric)

                        elif metric_choice == '4':
                          metric = max_prob_scores.cpu()
                          fpr, tpr, thresholds = roc_curve(comparisson.cpu(), metric)
                        
                        print("ola3")
                 
                        # Calculate the AUC
                        roc_auc = auc(fpr, tpr)

                        # Plot the ROC curve
                        plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, lw=2)
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic (ROC) Curve for FFNN-type SUNN')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig('roc_curve.png')
                        plt.show()

                        # Print the AUC
                        print("ROC AUC:", roc_auc)

                        # Optionally, if you still want to calculate confusion matrix at a specific threshold
                        binary_predictions = [1 if p >= threshold else 0 for p in metric]

                        # Calculate the confusion matrix
                        confusion = confusion_matrix(comparisson.cpu(), binary_predictions)
                        print(confusion)

                        # Define class labels for your problem
                        class_labels = ["Negative", "Positive"]  # Horizontal labels
                        # Modify class_labels based on your actual class labels

                        # Define threshold value in scientific notation
                        threshold_scientific = format(threshold, ".2e")

                        # Create a custom colormap where everything is grey
                        cmap = plt.matplotlib.colors.ListedColormap(['grey', 'grey', 'grey', 'grey'])

                        # Create a confusion matrix plot without the color scale
                        plt.figure(figsize=(6, 4))
                        plt.imshow(confusion, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
                        plt.title(f'Confusion Matrix for Threshold={threshold_scientific}')
                        tick_marks = np.arange(len(class_labels))
                        plt.xticks(tick_marks, class_labels)  # Horizontal class labels
                        plt.yticks(tick_marks, class_labels)  # Horizontal class labels
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        for i in range(len(class_labels)):
                            for j in range(len(class_labels)):
                                text_color = "black" if confusion[i, j] == 0 else "white"
                                plt.text(j, i, format(confusion[i, j], 'd'), horizontalalignment="center", color=text_color)
                        plt.savefig('cm.png')
                        plt.show()

                elif experiment_choice == "2":
                    print("ola") 
                    if curve_type_choice == '1': #PR curve
                        if metric_choice == '1':
                          metric = noisy_entropy_scores.cpu()
                          precision, recall, thresholds = precision_recall_curve(noisy_comparisson.cpu(), metric)

                        elif metric_choice == '2':
                          metric = noisy_mi_scores.cpu()
                          precision, recall, thresholds = precision_recall_curve(noisy_comparisson.cpu(), metric)

                        elif metric_choice == '3':
                          metric = noisy_jsd_scores.cpu()
                          precision, recall, thresholds = precision_recall_curve(noisy_comparisson.cpu(), metric)

                        elif metric_choice == '4':
                          metric = noisy_max_prob_scores.cpu()
                          precision, recall, thresholds = precision_recall_curve(noisy_comparisson.cpu(), metric)
                
                        # Calculate the AUPR
                        aupr = auc(recall, precision)

                        # Plot the PR curve
                        plt.figure(figsize=(8, 6))
                        plt.plot(recall, precision, lw=2)
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title('Precision-Recall Curve for FFNN-type SUNN')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig('precision_recall_curve.png')
                        plt.show()

                        # Print the AUPR
                        print("AUPR:", aupr)
                        
                        binary_predictions = [1 if p >= threshold else 0 for p in metric]
                        confusion = confusion_matrix(noisy_comparisson.cpu(), binary_predictions)
                        print(confusion)

                        cmap = plt.matplotlib.colors.ListedColormap(['grey', 'grey', 'grey', 'grey'])
                        class_labels = ["Negative", "Positive"]  # Horizontal labels

                        # Create a confusion matrix plot without the color scale
                        plt.figure(figsize=(6, 4))
                        plt.imshow(confusion, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
                        plt.title(f'Confusion Matrix for Threshold={threshold}')
                        tick_marks = np.arange(len(class_labels))
                        plt.xticks(tick_marks, class_labels)  # Horizontal class labels
                        plt.yticks(tick_marks, class_labels)  # Horizontal class labels
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        for i in range(len(class_labels)):
                            for j in range(len(class_labels)):
                                text_color = "black" if confusion[i, j] == 0 else "white"
                                plt.text(j, i, format(confusion[i, j], 'd'), horizontalalignment="center", color=text_color)
                        plt.savefig('cm.png')
                        plt.show()


                    elif curve_type_choice == '2': #Roc curve

                        if metric_choice == '1':
                          metric = noisy_entropy_scores.cpu()
                          fpr, tpr, thresholds = roc_curve(noisy_comparisson.cpu(), metric)

                        elif metric_choice == '2':
                          metric = noisy_mi_scores.cpu()
                          fpr, tpr, thresholds = roc_curve(noisy_comparisson.cpu(), metric)

                        elif metric_choice == '3':
                          metric = noisy_jsd_scores.cpu()
                          fpr, tpr, thresholds = roc_curve(noisy_comparisson.cpu(), metric)

                        elif metric_choice == '4':
                          metric = noisy_max_prob_scores.cpu()
                          fpr, tpr, thresholds = roc_curve(noisy_comparisson.cpu(), metric)

                        # Calculate the AUC
                        roc_auc = auc(fpr, tpr)

                        # Plot the ROC curve
                        plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, lw=2)
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic (ROC) Curve for FFNN-type SUNN')
                        plt.grid(True)
                        print("ROC AUC:", roc_auc)

                        # Optionally, if you still want to calculate confusion matrix at a specific threshold
                        binary_predictions = [1 if p >= threshold else 0 for p in metric]

                        # Calculate the confusion matrix
                        confusion = confusion_matrix(noisy_comparisson.cpu(), binary_predictions)
                        print(confusion)

                        # Define class labels for your problem
                        class_labels = ["Negative", "Positive"]  # Horizontal labels
                        # Modify class_labels based on your actual class labels

                        # Define threshold value in scientific notation
                        threshold_scientific = format(threshold, ".2e")

                        # Create a custom colormap where everything is grey
                        cmap = plt.matplotlib.colors.ListedColormap(['grey', 'grey', 'grey', 'grey'])

                        # Create a confusion matrix plot without the color scale
                        plt.figure(figsize=(6, 4))
                        plt.imshow(confusion, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
                        plt.title(f'Confusion Matrix for Threshold={threshold_scientific}')
                        tick_marks = np.arange(len(class_labels))
                        plt.xticks(tick_marks, class_labels)  # Horizontal class labels
                        plt.yticks(tick_marks, class_labels)  # Horizontal class labels
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        for i in range(len(class_labels)):
                            for j in range(len(class_labels)):
                                text_color = "black" if confusion[i, j] == 0 else "white"
                                plt.text(j, i, format(confusion[i, j], 'd'), horizontalalignment="center", color=text_color)
                        plt.savefig('cm.png')
                        plt.show()




            else:
                print("Invalid metric choice. Please enter a valid option (1, 2, 3, or 4).")
        else:
            print("Invalid curve type choice. Please enter 1 for PR Curve or 2 for ROC Curve.")
    else:
        print("Invalid experiment choice. Please enter 1 for Misclassification Experiment or 2 for OOD Detection Experiment.")


if __name__ == '__main__':

    tensor_list_logits_path = sys.argv[1]
    tensor_list_noisy_logits_50_path = sys.argv[2]
    test_labels_path = sys.argv[3]
  
    tensor_list_logits = torch.load(tensor_list_logits_path)
    test_labels = torch.load(test_labels_path).to("cuda")
    tensor_list_noisy_logits_50 = torch.load(tensor_list_noisy_logits_50_path)

    main(tensor_list_logits, tensor_list_noisy_logits_50, test_labels.to("cuda"))	