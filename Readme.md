# Image and NIR Spectral Data Fusion for Skin Cancer Classification

This repository contains the code used in the article [Insert article], providing scripts and tools for the analysis and classification of skin lesions using deep learning models.

## Features

- Implementations of Convolutional Neural Networks (CNNs) for skin cancer image analysis

- Classification and training of an MLP neural network

- Calculation and visualization of SHAP values

## Dependencies
The code is developed in Python and uses the following main libraries:

- PyTorch: For implementing deep learning models.

- Scikit-Learn: For complementary machine learning utilities.

To set up the environment, make sure Python is installed on your system. Then install the required dependencies using:

```
pip install -r requirements.txt
```


## Repository Structure

- class: Contains CNN model implementations and scripts for running experiments.

- shap: Script for the SHAP structure.

- inputs: Contains the folders for the datasets used.


##  Links to Datasets
The datasets used in this project are publicly available:

--- colocar o dataset [I+NIR-SC]


## Instructions for Use
Step 1: Clone the Repository

```
git clone https://github.com/BrunoSantosFF/I-NIR-SC.git
```

Step 2: Configure Dataset Paths

Inside the main file (**main.py**), in the MODEL_CONFIGS variable, configure the input files for each chosen model.

Step 3: Run Experiments

To run an experiment, simply use the main.py script with the following command. An interactive menu will appear: 

```
python3 main.py 
```

## Contact
For any questions, please contact me at brunosf1804@gmail.com

## Credits, Acknowledgments, and Notes
Some code implementations were based on the following repositories, available at: https://github.com/paaatcha/my-thesis and https://github.com/MatheusBecali/VAE-GNA.git

These adaptations were essential for:

- Ensuring the reproducibility of results previously obtained with the PAD-UFES-20 dataset, allowing fine-tuning with the I+NIR-SC dataset and saving its features for use in the code.

- The same procedure was performed on the NIR-SC-UFES dataset to obtain the latent vector for the inputs.



