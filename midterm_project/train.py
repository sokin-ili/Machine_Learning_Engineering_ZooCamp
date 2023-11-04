################################## >>>      IMPORT PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
import pickle

from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# import lightgbm as lgb
# from sklearn.svm import SVC
from IPython.display import display

# import os
# os.chdir("C:/Users/sokin/Documents/Zoocamp/midterm_project")

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, classification_report, precision_score, recall_score, accuracy_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
# sns.set_theme()
print('\n 1/6. IMPORT PACKAGES done! \n')


################################## >>>      INITIAL DATA MANIPULATION
df = pd.read_csv('data.csv')
df.columns = df.columns.str.lower()

df.diagnosis = (df.diagnosis == 'M').astype('int')
df = df.drop(['id', 'unnamed: 32'], axis=1)
print('\n 2/6. INITIAL DATA MANIPULATION done! \n')

################################## >>>      TRAIN, TEST, FULL TRAIN SPLIT 
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

y_train = df_train.diagnosis.values
y_val = df_val.diagnosis.values
y_test = df_test.diagnosis.values

del df_train['diagnosis']
del df_val['diagnosis']
del df_test['diagnosis']
print('\n 3/6. TRAIN, TEST, FULL TRAIN SPLIT  done! \n')


################################## >>>       FEATURE ENGINEERING 
from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(df_train_full.values)
scaled_features_train_full = pd.DataFrame(scaled_features, index=df_train_full.index, columns=df_train_full.columns)

# Highly correlated features are discarded! 
high_cor_feat = [
    'radius_mean', 'perimeter_mean', 'radius_se', 
    'radius_worst', 'perimeter_worst','perimeter_se', 'texture_mean'
]
df_train = df_train.drop(high_cor_feat, axis=1)
df_test = df_test.drop(high_cor_feat, axis=1)
df_val = df_val.drop(high_cor_feat, axis=1)

# rfecv features
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
X_train = df_train.values
X_val = df_val.values

print('X_val,X_train created for RFECV')
rf = RandomForestClassifier() 
rfecv = RFECV(estimator=rf, step=1, cv=7, scoring='roc_auc')   # 10-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', df_train.columns[rfecv.support_])
rfecv_feat = list(df_train.columns[rfecv.support_])
print('\n 4/6. FEATURE ENGINEERING  done! \n')

print(rfecv_feat)

################################## >>>      MODELING 
sc = StandardScaler()
scaled = sc.fit_transform(df_train_full[rfecv_feat])
scaled_df_train_full = pd.DataFrame(scaled, index=df_train_full.index, columns=df_train_full[rfecv_feat].columns)
print('data standard scaled')
X_train_full, y_train_full = scaled_df_train_full[rfecv_feat].values, df_train_full.diagnosis.values

scaled = sc.transform(df_test[rfecv_feat])
scaled_df_test = pd.DataFrame(scaled, index=df_test.index, columns=df_test[rfecv_feat].columns)

# Logistic Regression with hyper-tuning params
model =LogisticRegression(C= 0.05, penalty= 'l2', solver= 'liblinear', random_state=42) # hyper-tuned !
model.fit(X_train_full, y_train_full)
print('model Logistic Regression trained')

y_pred = model.predict(scaled_df_test)
print('AUC on test set:', np.round(roc_auc_score(y_pred, y_test), 6))
print('AUC on full train set:', 0.97013) #auc score obtained from notebook.ipynb
print('Recall score:', np.round(recall_score(y_pred, y_test), 6))

print('\n 5/6. MODELING done! \n')

################################## >>>      EXPORT MODEL
c = 0.05
output_file = f'model_LogReg_C={c}.bin' 

# Export
f_out = open(output_file, 'wb')
pickle.dump((sc, model), f_out)
f_out.close()
print('\n 6/6. MODEL EXPORT done! \n')