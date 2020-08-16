# Tox21 dataset challenge

## 1.Package installation in Google CO Lab envirment

in this cell we collect all the required package installation in our project, It is like batsh installation in linux envirment, almost all of the packages are installed using " pip ", some others use " conda "  or imported from GitHub.
```python
# Installing scikit learn
!pip install scikit-learn

# installing scikit-multilearn
'''
!pip install scikit-multilearn


#Installing rdkit

!wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!time bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!time conda install -q -y -c conda-forge rdkit


# Molecular descriptor calculator package
# https://github.com/mordred-descriptor/mordred/blob/develop/README.rst

!pip install 'mordred[full]'


#installing seaborn 
!pip install --upgrade seaborn==0.9.0

#installing SDF
!pip install SDF

#installing  
!pip install wget

#add the new installed packages to the path 

%matplotlib inline
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/usr/local/lib/python3.7/site-packages')
```

### 1.1. Import Packages, Libraries and Frameworks 

```python
import numpy as np
import tensorflow as tf 
import sklearn as sk
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split as train_test_split 



import seaborn as sns
import warnings
#import rdkit as rd
#from rdkit import Chem

# configuration of tf 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.keras.backend.set_session(tf.Session(config=config))
```
### 1.2. Install the PyDrive wrapper & import libraries.
we use pyDrive to import data or files from google drive in colab envirment.

```python
# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
#https://drive.google.com/open?id=1pAZqQhRgzE4j59DnFZdrRRDOP91us7LD
#https://drive.google.com/open?id=14puiaHdD_-VMTA7XTMkJS3AkpAaqQymU
#https://drive.google.com/open?id=1gfYF3FYaoegrJJ08sU57AXtVR8aUULVF
#https://drive.google.com/open?id=1581t_2gKiptUcXIh-kbZ2uDlSI2iu-jQ

#new Cleaned X data : Nan Feature sets removed
#Train : https://drive.google.com/open?id=1EfC---GqsIObD0ouhJnSVTzSCkUPHMSj
#Test : https://drive.google.com/open?id=1qfqpgnSJFjoOBfv_zpRi40Wf7PIzXFgo


fileid1 = '1EfC---GqsIObD0ouhJnSVTzSCkUPHMSj'
fileid2 = '1qfqpgnSJFjoOBfv_zpRi40Wf7PIzXFgo'
fileid3 = '1pAZqQhRgzE4j59DnFZdrRRDOP91us7LD'
fileid4 = '14puiaHdD_-VMTA7XTMkJS3AkpAaqQymU'

downloaded1 = drive.CreateFile({'id' : fileid1 })
downloaded2 = drive.CreateFile({'id' : fileid2 })
downloaded3 = drive.CreateFile({'id' : fileid3 })
downloaded4 = drive.CreateFile({'id' : fileid4 })
downloaded1.GetContentFile("tox21_descriptors_5k_train")
downloaded2.GetContentFile("tox21_descriptors_5k_test")
downloaded3.GetContentFile("tox21_10k_data_all")
downloaded4.GetContentFile("tox21_10k_challenge_test")

#print('Downloaded content "{}"'.format(downloaded.GetContentString()))
```
## 2.Data Preprocessing 

### Import orginal tox21 data 

The original file of tox21 is an sdf file contains a table of 17 columns and  12K lines.

#### Process used for Tox21 Data preparation an preprocessing
in data preparation phase we follow many rules to prepare the data, they are necessary before start building the model, there are many process but in our case we will use just some of them:
   -  Convert the data from raw data to numeric.
   -  clean the data 
      - filling missing values 
   - Data transformation   
   - Standardization of data 
   -  Spliting the data to train and test sets
   in fact, Tox21 data has been splited before by the challenge organizers to evalute the results of participants, but we can resplit it randomly to look for deferent results.

```python
#extarct file to directory 

import zipfile
import gzip
import wget

#tox21 dataset original sourse link
url='https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf&sec='  

wget.download(url, '/content/')  

zip_ref = zipfile.ZipFile("../content/tox21_10k_data_all.sdf.zip", 'r')
zip_ref.extractall("../content/")
zip_ref.close()

```

### 2.1. Loading the data from the SDF file 

We use Pandas tools from rdkit library to load the data of sdf format to a Pandas Dataframe object.''
```python
from rdkit.Chem import PandasTools

#LOAD THE training DATA FROM SDF FILE 
path = "../content/tox21_10k_data_all" 
tox21_10k_challenge= PandasTools.LoadSDF(path)
# Load the test data from sdf file 
path2= "../content/tox21_10k_challenge_test"
tox21_10k_challenge_test=PandasTools.LoadSDF(path2)

#Displaing the first four lines of the Data 
tox21_10k_challenge.head()
```
### 2.2. Exporting the data to Excel file 
```python
print(tox21_10k_challenge.columns)
tox21_10k_challenge.to_excel('Tox21_10k_data_all.xlsx',columns= ('DSSTox_CID', 'FW', 
'Formula', 'ID', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 
'SR-p53'))

tox21_10k_challenge_test.to_excel('Tox21_10k_challenge_test.xlsx',columns= ('DSSTox_CID', 'FW', 
'Formula', 'ID', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 
'SR-p53'))
```
### 2.3. Loading the exported execl file to prepare the data 

We put the data in a pandas dataframe after that we delete the columns that will not be used in the computing process, we keep only the columns of the 12 tasks wich represent the labels of this dataset. this labels will be used in the predictive model.
```python
#loading Execl data of tox21 Train and test sets into Pandas data frame

tox21_10k_train=pd.read_excel('../content/Tox21_10k_data_all.xlsx')
tox21_10k_test=pd.read_excel('../content/Tox21_10k_challenge_test.xlsx')

#Prepaering and cleansing the data

## Deleting the ids, smile formula columns 
tox21_10k_train.drop(['Unnamed: 0','DSSTox_CID','FW','Formula','ID'],axis=1, 
                                                                  inplace=True)
tox21_10k_test.drop(['Unnamed: 0', 'DSSTox_CID','FW','Formula','ID'],axis=1, 
                                                                  inplace=True)

## fill Nan columns with zeros

tox21_10k_train=tox21_10k_train.fillna(0)
tox21_10k_test=tox21_10k_test.fillna(0)
tox21_10k_train.head
```
### 2.4. Loadoing the calculated descriptors data and prepare it for the processing phase.

The descriptors data represents the examples ( set of features X ) that will be used in the predictive model 
```python
# Read the Dataset from Exel files
xlxsf1=pd.ExcelFile('../content/tox21_descriptors_5k_train')
xlxsf2=pd.ExcelFile('../content/tox21_descriptors_5k_test') 
#print(xlxsf1.sheet_names)# 'tox21_descriptors_5k_train'
#print(xlxsf2.sheet_names)# 'Sheet2'
tox21_desc_train= xlxsf1.parse(sheet_name='Sheet2')
tox21_desc_test= xlxsf2.parse(sheet_name='Sheet2')

print(tox21_desc_train.columns)
print(tox21_desc_test.columns)

# Read the Dataset from a csv file 
#tox21_desc_train= pd.read_csv('../content/tox21_descriptors_5k_train', delimiter=';')
#tox21_desc_test= pd.read_csv('../content/tox21_descriptors_5k_test', delimiter=';')


# drop 'No.' and 'NAME' columns

tox21_desc_train.drop(['No.','NAME','Uc'], axis=1, inplace=True)
tox21_desc_test.drop(['No.','NAME'], axis=1, inplace=True)


#dropoing Nan columns 
# to drop comumns which all their features are Nan
# Pandas has problems in this built-in, 

#tox21_desc_train.dropna(axis='columns', how='all')
#tox21_desc_test.dropna(axis='columns', how='all')

# filling Nan cells with Zeros 

    # Get indices of NaN
    #inds1 = pd.isnull(tox21_desc_train).all(1).nonzero()[0]
    #inds2 = pd.isnull(tox21_desc_test).all(1).nonzero()[0]
    #print(inds1)
    #print(inds2)

tox21_desc_train.replace('Nan', 0 , inplace=True)
tox21_desc_test.replace('Nan', 0 ,inplace=True)

tox21_desc_train.fillna(0, inplace=True)
tox21_desc_test.fillna(0,inplace=True)

#Show the data

print(tox21_desc_train.head())
print(tox21_desc_test.head())



#inds1 = pd.isnull(tox21_desc_train).all(1)
#inds2 = pd.isnull(tox21_desc_test).all(1)

print(tox21_desc_test.shape)
print(tox21_desc_train.shape)


#convert the data to numeric 

#def coerce_to_numeric(df, column_list):
#    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
#coerce_to_numeric(tox21_descriptors_df, tox21_descriptors_df.columns)
#

#changing the data type to float 
#
tox21_10k_train_T = np.array(tox21_10k_train.values, dtype = np.float64)
tox21_10k_test_T = np.array(tox21_10k_test.values, dtype = np.float64)
tox21_desc_train_T = np.array(tox21_desc_train.values, dtype = np.float64)
tox21_desc_test_T = np.array(tox21_desc_test.values, dtype=np.float64)
```

### 2.5. Standardization of Data 

The data has been scalled using  Scikit learn ***StandarScalar*** function to standardize features by removing the mean and scaling to unit variance

The standard score of a sample x is calculated as:
 $  z = (x - u) / s $
where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

```python
#standardization: scaling the data 

scaler = sk.preprocessing.StandardScaler()

scaler.fit(tox21_desc_train_T)
scaler.fit(tox21_desc_test_T)

# transforming the data

tox21_desc_train_S = scaler.transform(tox21_desc_train_T)
tox21_desc_test_S = scaler.transform(tox21_desc_test_T)

'''tox21_descriptors_df = pd.DataFrame(scaler.transform(tox21_descriptors_d), 
columns=tox21_descriptors_df.columns)'''
```

### 2.6. Spliting the data 
we have two spliting strategy :
 #### 1 - the default spliting  
as we mention before, the organizers have prepared the data for the participants to evaluate them.
 #### 2 - Random Split 
 
The data splited randomly into two sets : training set and test set , using scikit learn ***train_test_split*** module , we give the training set 75% of the data set and the test set about 25%.

see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

```python
# changing the variable 
x_train = tox21_desc_train_S
y_train = tox21_10k_train_T

x_test = tox21_desc_test_S
y_test = tox21_10k_test_T

# checking the shape of data before building the model 
# reshape the x train data 
x_train = np.delete(x_train, [11758,11759,11760, 11760, 11761, 11762, 11763, 11764], axis=0)
x_test = np.delete(x_test , [295] , axis= 0 )

#print('Tox21 dataset shape after cleansing ',printtox21_m.shape)

# SPLIT 290/11764

x_train, y_train,x_val, y_val = train_test_split( x_train , y_train, test_size=0.02465)
val_data= (x_val,y_val)

#reshaping the traing set 
#x_train =x_train.reshape(x_train.shape[0],x_train.shape[1])
#x_test =x_test.reshape(x_test.shape[0], x_test.shape[1] )

print('y_train.shape =', y_train.shape)
print('x_train.shape =', x_train.shape)

print('x_val.shape =', x_val.shape)
print('y_val.shape =' , y_val.shape)

print('x_test.shape =', x_test.shape)
print('y_test.shape =' , y_test.shape)

# 3.Building the Deep Neural Network model 

```
### 3.1. Model Configuration  parameters and Hyperarameters configuration

#### 3.1.1. Network Architecture:
    - the 1st hidden layers 
     - Number of neurons = 500
     - Activation function : relu 
    - the 2nd hidden layers 
      - Number of neurons = 500
      - Activation function : relu   
    - the 3rd : Output layer 
      - Activation : Softmax  , SIGMOID
      - Number of neurons = 12 (12 classes) multiclassifaction problem
#### 3.1.2. The optimizer and loss function: 
    - adam optimizer
    - batch size = 128 , 64
    - lr=0.001 
    - beta_1=0.9 
    - beta_2=0.999
    - #epochs = 100 , 200

#### 3.1.3. Loss function
    -  binary cross-entropy ( rather than categorical cross-entropy.)
    This may seem counterintuitive for multi-label classification; however, the goal is to treat each output label as an independent Bernoulli distribution and we want to penalize each output node independently.
    
#### 3.1.4. The metrics used to evaluate the results     
    - ROC_AUC
    - binary accuracy 
#### 3.1.5.Regularization 
    - Regularization techniques used to minimise the high variance (reduce overfiting): 
     - L2 regularization with 0.0001
     - dropout (inverted dropout)
     - Augmented data : generate new features from AlvDesc 0.1 software which contains more than 7000 chemical descriptors.
```python
#optimzers
adam = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999)
adammax =tf.keras.optimizers.Adamax()
sgd=tf.keras.optimizers.SGD()
#losses
bce=tf.keras.losses.binary_crossentropy
mse=tf.keras.losses.mean_squared_error
#Metrics
#b_acc=tf.keras.metrics.binary_accuracy(y_true, y_pred)


model.compile( optimizer=adam , loss =bce, metrics= ['binary_accuracy'])
```
```python

from keras import regularizers
# The number of hidden neurons should be between the size of the input layer and 

#
model=tf.keras.models.Sequential( [
      tf.keras.layers.Dense(500, kernel_regularizer= tf.keras.regularizers.l2(0.0005), activation = "relu" ),
#     tf.keras.layers.Dropout(0.5),  
      tf.keras.layers.Dense(500, kernel_regularizer= tf.keras.regularizers.l2(0.0005),activation = "relu" ),
#     tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(200, kernel_regularizer= tf.keras.regularizers.l2(0.0005),activation = "relu" ),
#     tf.keras.layers.Dropout(0.5),    
      tf.keras.layers.Dense(12, activation = "softmax")])
```
```python
fit_log = model.fit(x_train, y_train , epochs=100, batch_size=64, validation_data=val_data) #batch_size=128
```
```python
#print(model.get_config())
test_loss = model.evaluate(x_test, y_test)
```python
print(model.summary())
```
```python
y_pred = model.predict(x_test).ravel()
y_pred_v = model.predict(x_val).ravel()
y_pred_tr = model.predict(x_train).ravel()

y_test = y_test.ravel()
y_val = y_val.ravel()
y_train = y_train.ravel()

print(y_test.shape)
print(y_pred.shape)
print(y_train.shape)
print(y_pred_tr.shape)
print(y_pred_v.shape)
print(y_val.shape)
```

### ROC - AUC metric 

Reference of code : https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#ROC 
fpr_t, tpr_t, thresholds_t = roc_curve(y_test, y_pred)
fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_pred_v)
fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train, y_pred_tr)

# AUC 
auc_test = auc(fpr_t, tpr_t)
auc_val = auc(fpr_val, tpr_val)
auc_tr = auc(fpr_tr, tpr_tr)
print('AUC=', auc_test)
print('AUC=', auc_val)
print('AUC=', auc_tr)
# plot 

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_t, tpr_t, label='(auc_test area = {:.3f})'.format(auc_test))
plt.plot(fpr_val, tpr_val, label='(auc_val area = {:.3f})'.format(auc_val))
plt.plot(fpr_tr, tpr_tr, label='(auc_tr area = {:.3f})'.format(auc_tr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.show()

plt.savefig("AUC.pdf")
plt.savefig("AUC.png")
```

### Training and validation Loss curves
```python
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.ylim(0.,1.2)
plt.plot( fit_log.epoch, fit_log.history["loss"], label="Train loss")
plt.plot( fit_log.epoch, fit_log.history["val_loss"], label="Valid loss")
plt.legend()
# 
plt.savefig("Loss.pdf")
plt.savefig("Loss.png")
```

### Accuracy Curve 
```python
plt.xlabel('epochs')
plt.ylabel('binary_accuracy')
plt.title('binary_accuracy curve')
plt.ylim(0.9,1.)
plt.plot(fit_log.history['binary_accuracy'], label='train')
plt.plot(fit_log.history['val_binary_accuracy'], label='Valid')
plt.legend()
#plt.show()
plt.savefig("accuracy.pdf")
plt.savefig("accuracy.png")
```
```python
fit_log.history.keys()
```

### Save  the model and results in google drive 
```python
from keras.models import load_model
from datetime import datetime

now = datetime.now()
today = now.strftime("%d-%m-%Y %H:%M:%S")
print(today)
model.save('Model '+today+'.h5')
#my_model = load_model('my_model.h5')

#uploading file to Google drive from colab VM

#code reference : https://gist.github.com/yt114/dc5d2fd4437f858bb73e38f0aba362c7 

!pip install -U -q PyDrive
!git clone https://github.com/Joshua1989/python_scientific_computing.git
!git clone https://gist.github.com/dc7e60aa487430ea704a8cb3f2c5d6a6.git /tmp/colab_util_repo
!mv /tmp/colab_util_repo/colab_util.py colab_util.py 
!rm -r /tmp/colab_util_repo
```


### Create forlder and subforlders to save models in Google drive 
```python
from colab_util import *
drive_handler = GoogleDriveHandler()

# creating folder and subfolders in google drive 

#test_folder_id = drive_handler.create_folder('Tox21 Models and Results')
#test_folder_id
#same_subfolder_id = drive_handler.create_folder('Models', parent_path='Tox21 
#Models and Results')

# create subfolder to save the current model 

subfolder_id = drive_handler.create_folder('Model '+ today, parent_path='Tox21 '
+'Models and Results/Models')

# uploading the model, plots, to a drive folder
model_path='../content/Model '+today+'.h5'
auc_path='../content/AUC.png'
loss_path='../content/Loss.png'
acc_path='../content/accuracy.png'
up={model_path , auc_path , loss_path , acc_path}
for i in up :
  drive_handler.upload(i , parent_path='Tox21 Models and Results/Models/Model '
+today)
```
## References
```python
1 - https://github.com/jupyter/nbconvert/issues/314
2 - https://github.com/jupyter/nbconvert/issues/524
3 - https://www.pyimagesearch.com/2018/05/07multi-label-classification-with-keras/ 
```
