
# Flower image classification

As a part of my last capstone project for the Machine Learning Engineering Zoomcamp, I decided to focus on the application of Convolutional Neural Networks (CNNs). I followed the steps of section 8.Machine Learning Zoomcamp(https://www.youtube.com/watch?v=it1Lu7NmMpw&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=67). Furthermore, the dataset contains 1600 images for every type(bellflower, daisy, dandelion, lotus, rose, sunflower, tulip). Flower image classification is important:

1. Biodiversity Monitoring\
Enables automated identification and tracking of different flower species in the wild, while contributing to biodiversity monitoring efforts by providing valuable data on the distribution and abundance of plant species.

2. Ecosystem Health Assessment\
Offers a non-intrusive method for assessing the health of ecosystems based on the diversity and condition of flowering plants.
Provides insights into the overall well-being of ecosystems and potential ecological imbalances.

3. Educational Tools\
Serves as an educational resource for botany students, researchers, and nature enthusiasts. Moreover, it facilitates the understanding of plant diversity and characteristics.

4. Automated Floriculture Industry\
 **Benefits the floriculture industry by automating the sorting and categorization of flowers based on species and characteristics. In addition, it enhances efficiency and reduces manual labor in the production and sale of flowers.**


# Data

You can download the data from my *Google Drive*: https://drive.google.com/drive/folders/18e6XK3vwEmHCzW8uKrHmsRufEqRxht35?usp=sharing or you can find a smaller dataset in *Kaggle*: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition [**has fewer categories and images**]
 

# CNNs

You can download the trained models from my *Google Drive*: https://drive.google.com/drive/folders/18e6XK3vwEmHCzW8uKrHmsRufEqRxht35?usp=sharing . Each file is ~250 MB, thus they were too much for Github :P

# Instructions 

First of all, you should create & activate the project's environment (in folder named env, the environment's name is: **cnn_mle**). There you will also find the requirements.txt file with every module you will need.

Download the repo and in the directory(breast_cancer_classification) activate the environment (change the *user*). You might need to download *virtualvenv*.

```bash
pip install virtualenv

mkdir flower_image_classification

cd flower_image_classification

mkdir env

cd env

python<version> -m venv cnn_mle

```
(for Unix: **python3 -m venv cnn_mle**)

On Unix or MacOS, using the bash shell: source /path/to/venv/bin/activate\
On Windows using the Command Prompt: path\to\venv\Scripts\activate.bat

```bash
source /home/user/flower_image_classification/env/cnn_mle/bin/activate
```
In case you are using Jupyter notebooks or you have any error following the previous steps, just open the requirements.txt and in anaconda cmd enter manually *conda install <package>* for every package. For instance,

```bash
conda install pandas
```

The notebook has 8 sections:\
    1. Import libraries \
    2. Train a smaller image size locally \
    3. Import xception model + checkpoint + compile \
    4. Train model \
    5. Model Accuracy & Loss \
    6. Evaluate model_cnn  \
    7. Load model_xception and evaluate\

In section 7, I trained a more complex CNN based on xception. Since my resources were limited I decided to train the CNN in Google Colab. For this action, you can run the flowers_notebook in Google Colab but it is necessary to run also the following in a cell

```bash
from google.colab import drive
drive.mount('/content/drive')
```
this will import data from your google drive into Google Colab! **No deployment was applied!**


# Baseline model: model_cnn

*Train Accuracy*: 88.53%
*Validation Accuracy*: 84.42%

# Final model: model_xception

*Train Accuracy*: 99.53%
*Validation Accuracy*: 94.20%

## 🔗 Links
If you have any questions or feedback, please reach out..

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nikos-iliopoulos-186a58157/)
