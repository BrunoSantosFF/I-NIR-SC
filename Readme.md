# Multimodal Fusion of Skin Lesion Images and NIR Spectroscopy: A Novel Dataset for Classification Evaluation

# Under REVIEW

This repository contains the code used in the article [Multimodal Fusion of Skin Lesion Images and NIR
Spectroscopy: A Novel Dataset for Classification Evaluation](#features), providing scripts and tools for the analysis and classification of skin lesions using deep learning models.

## Features

- Implementations of Convolutional Neural Networks (CNNs) for skin cancer image analysis

- Classification and training of an MLP neural network

- Calculation and visualization of SHAP values

## Dependencies
The code is developed in Python and uses the following main libraries:

- PyTorch: Deep learning modeling and training.
- Scikit-learn: Auxiliary machine learning utilities.

To set up the environment, make sure Python is installed on your system. Then install the required dependencies using:

```bash
pip install -r requirements.txt
```


## Repository Structure

- `class/`: CNN model implementations and experiment scripts.
- `shap/`: Scripts used for SHAP analysis.
- `inputs/`: Input files and auxiliary data used by experiments.

```text
.
├── class/
│   ├── data_loader.py
│   ├── main.py
│   ├── model.py
│   └── plot.py
├── inputs/
│   └── vetor_latente.txt
├── shap/
│   └── shap.py
└── requirements.txt
```


## Links to Datasets
The datasets used in this project are publicly available:

- I+NIR-SC-UFES: [Mendeley Data](https://data.mendeley.com/datasets/fttnv8x2t5/1)


## Instructions for Use
Step 1: Clone the Repository

```bash
git clone https://github.com/BrunoSantosFF/I-NIR-SC.git
```

Step 2: Configure Dataset Paths

Inside the main file (**main.py**), in the MODEL_CONFIGS variable, configure the input files for each chosen model.

Step 3: Run Experiments

To run an experiment, simply use the main.py script with the following command. An interactive menu will appear: 

```bash
python3 main.py 
```

## Contact
For any questions, please contact me at brunosf1804@gmail.com

## Credits, Acknowledgments, and Notes
Some code implementations were based on the following repositories, available at: https://github.com/paaatcha/my-thesis and https://github.com/MatheusBecali/VAE-GNA.git

These adaptations were essential for:

- Ensuring the reproducibility of results previously obtained with the PAD-UFES-20 dataset, allowing fine-tuning with the I+NIR-SC-UFES dataset and saving its features for use in the code.

- The same procedure was performed on the NIR-SC-UFES dataset to obtain the latent vector for the inputs.



