
# Breast Cancer Classification

Breast cancer is cancer that forms in the cells of the breasts. It can occur in both men and women, but it's far more common in women.

Substantial support for breast cancer awareness and research funding has helped create advances in the diagnosis and treatment of breast cancer. Breast cancer survival rates have increased, and the number of deaths associated with this disease is steadily declining, largely due to factors such as **earlier detection**, a new personalized approach to treatment and a better understanding of the disease.

This project aims to ease the job of malignant/benign cancer identification from medical personnel, using machine learning algorithms to classify the cancer samples! **This procedure will result in more accurate and faster results, thus minimizing time for discovering if cancer is malignant/benign**. Breast cancer classification divides breast cancer into categories according to different schemes criteria and serving a different purpose. In addition, the classification of breast cancer is usually, but not always, primarily based on the histological appearance of tissue in the tumor. 

The dataset for this project is from Breast Cancer Wisconsin (Diagnostic).

# Data

You can download the data from *UCI*: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic or from *Kaggle*: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data 

A deeper analysis on the features and their information can be seen on the notebook 

# Instructions 

I have created a **cloud account** in Hetzner and all files for the project live there. The cloud runs Ubuntu 22.04

First of all, you should create & activate the project's environment (in folder named env, the environment's name is: **mle_venv**). There you will also find the requirements.txt file with every module you will need.

Download the repo and in the directory(breast_cancer_classification) activate the environment (change the *user*). You might need to download *virtualvenv*.

```bash
pip install virtualenv

mkdir breast_cancer_classification

cd breast_cancer_classification

mkdir env

cd env

python<version> -m venv mle_venv

```
(for Unix: **python3 -m venv mle_venv**)

On Unix or MacOS, using the bash shell: source /path/to/venv/bin/activate\
On Windows using the Command Prompt: path\to\venv\Scripts\activate.bat

```bash
source /home/user/breast_cancer_classification/env/mle_venv/bin/activate
```
In case you are using Jupyter notebooks or you have any error following the previous steps, just open the requirements.txt and in anaconda cmd enter manually *conda install <package>* for every package. For instance,

```bash
conda install pandas
```

The notebook has 8 sections:\
    1. Initial data preparation \
    2. Train, validation, test splits \
    3.  EDA \
    4. Baseline model \
    5. Feature Engineering/Selection \
    6.  Final model  \
    7. Export final model\
    8. App deployment

You should run section 8 only after you have initially runned predict.py script (the procedure is the same as in the section 5 of ml-zoomcamp https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment)

Repo also contains the train.py script, where the model training procedure is followed.

### EDA

Univariate, bivariate and multivariate analysis with a plethora of visualizations (histograms, swarmplots, heatmaps etc.)

### Baseline model

Trained LogisticRegression,
              DecisionTreeClassifier,
              ExtraTreeClassifier,
              RandomForestClassifier,
              SGDClassifier,
              SVC, 
              XGBClassifier,
              LGBMClassifier 

###  Feature Engineering/Selection

Tried different techniques(RFECV, RandomForest etc.) on standard scaled features

### Final model

Trained engineered and selected data on 3 scanarios(all features, RFECV features, most important according to RandomForest)
LogisticRegression excelled almost in all scenarios and was choosen over other models for its model interpretation factor.

Hyper-parameter tuned LogisticRegression: the results did not shown overfitting and the model is **stable**!

Except AUC score, I focused also on **Recall**, since I wanted the model to grasp False Negative results(the medical personel needs to know every actual malignant cancers) 

A final highlight of this project, is the fact that my observations from EDA were accurate and verified, since the intepretation of my final model proved my assumptions!

### Spoiler :P 

AUC on test set: 0.974206\
AUC on full train set: 0.97013\
Recall score: 0.97619

## 🔗 Links
If you have any feedback, please reach out 

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nikos-iliopoulos-186a58157/)