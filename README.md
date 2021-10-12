# DNA
The following repository contains interactive notebooks which can be used to replicate the work and ideas presented in Disrupting Adversarial Transferability in Deep Neural Networks. Trained Dual Neck Autoencoder models are also included.
By directory:
classifier_exp: Contains notebooks and results used in the paper to compare separate CNN classifiers and their attack transferability
  -parameter_classifiers.ipynb can be used to train parallel classifiers with different parametric distances in parallel.
  -decorrelated_classifiers.ipynb can be used to train parallel classifiers with the decorrelation mechanism.
  -plot_figures.ipynb is used to create figures from the classifier results.
  -Results from evaluating transferability are stored as .npz files. These can be overwritten by rerunning the scripts in paramter_classifiers and decorrelated_classifiers.
  
DNA_exp: Contains subdirectories cifar_exp and mnist_exp, which each contain notebooks for recreating the cifar and mnist experiments respectively.
  -train_DNA.ipynb: notebook for training and saving DNA model and accompanying classifier.
  -train_AE.ipynb: notebook for training and saving an equivalent autoencoder model and accompanying classifier.
  -eval_results.ipynb: notebook for adversarially evaluating the models saved in 'models' directory
  -models: directory that contains autoencoder (ae), DNA (dna), and their accompanying classifiers for evaluation. These models can be overwritten by with traing_DNA.ipynb and train_AE.ipynb.


Notes on the environment used:
torch version 1.8.1
torchvision version 0.2.1
numpy version 1.19.2
pandas version 1.1.3
seaborne version 0.11.0
