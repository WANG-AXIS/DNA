# DNA
The following repository contains code and notebooks which can be used to replicate the work and ideas presented in Disrupting Adversarial Transferability in Deep Neural Networks. Trained models are also included.  
By directory:

classifier_exp: Contains notebooks and results used in the paper to compare separate CNN classifiers and their attack transferability.  
  -train.py: main script for training regular and decorrelated models
  -train_dverge: modified script for training using simplified DVERGE methodology
  -attacks.py: file containing attack algorithms
  -models.py: file containing model architectures

DNA_exp: Contains subdirectories cifar_exp and mnist_exp, which each contain notebooks for recreating the cifar and mnist experiments respectively.  
  -train_DNA.ipynb: notebook for training and saving DNA model and accompanying classifier.  
  -train_AE.ipynb: notebook for training and saving an equivalent autoencoder model and accompanying classifier.  
  -eval_results.ipynb: notebook for adversarially evaluating the models saved in 'models' directory.  
  -models: directory that contains autoencoder (ae), DNA (dna), and their accompanying classifiers for evaluation. These models can be overwritten by with train_DNA.ipynb and train_AE.ipynb.  

The concept of DVERGE training is attributed to Yang, et al. https://arxiv.org/abs/2009.14720

Notes on the environment used:  
torch version 1.8.1  
torchvision version 0.2.1  
numpy version 1.19.2  
pandas version 1.1.3  
seaborne version 0.11.0  
