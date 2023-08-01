# Titanic_Survival
Predicting the Survival of Titanic Passengers using Machine Learning


``` python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
```


```python
# load data 
data = pd.read_csv('Titanic.csv')
data.head()
```


