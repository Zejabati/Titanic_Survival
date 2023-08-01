# Titanic_Survival_Prediction
Predicting the Survival of Titanic Passengers using Machine Learning

**1**
```
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

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm



```
# load data
data = pd.read_csv('Titanic.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```
print(data.describe())
print(data.shape)
print(data.columns)
```

           PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
    count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
    mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
    std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
    min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
    25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
    50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
    75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
    max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200
    
    [8 rows x 7 columns]
    (891, 12)
    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')


**2**


```
null_counts = data.isnull().sum()
#null
print(null_counts[null_counts > 0].sort_values(ascending=False))
#percent
print((null_counts[null_counts > 0].sort_values(ascending=False)/len(data))*100)

#Sex male1 female0
data['Sex'] = data['Sex'].replace(['female','male'],[0,1])

# Remove some features
data=data.drop(columns=['Name','PassengerId','Ticket','Cabin'])
data.head()
```

    Cabin       687
    Age         177
    Embarked      2
    dtype: int64
    Cabin       77.104377
    Age         19.865320
    Embarked     0.224467
    dtype: float64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```
f, ax = plt.subplots(figsize = [10,9])
sns.heatmap(data.corr(),linewidths = .5, annot = True, cmap = 'YlGnBu', square = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fcd0566d1d0>




    
![png](Titanic_Survival_Prediction_files/Titanic_Survival_Prediction_7_1.png)
    



```
# find most frequent Embarked value and store in variable
most_embarked = data.Embarked.value_counts().index[0]
# fill NaN with most_embarked value
data.Embarked = data.Embarked.fillna(most_embarked)
data.head()

# create dummy variables for categorical features(One Hot Encoding)
data = pd.get_dummies(data,columns=['Embarked'])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```
SingleImputer = IterativeImputer(missing_values=np.nan)
DfSingle = data.copy(deep=True)
DfSingle.iloc[:,:] = SingleImputer.fit_transform(DfSingle)

MeanImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
DfMean = data.copy(deep=True)
DfMean.iloc[:, :] = MeanImputer.fit_transform(DfMean)

IterativeImputer = IterativeImputer(missing_values=np.nan, sample_posterior=True, min_value=0,random_state=0)
DfIterative = data.copy(deep=True)
DfIterative.iloc[:, :] = IterativeImputer.fit_transform(DfIterative)

data['Age'].plot(kind='kde', c='red')
DfMean['Age'].plot(kind='kde')
DfIterative['Age'].plot(kind='kde', c='yellow')
DfSingle['Age'].plot(kind='kde', c='green')
labels = ['Baseline (Initial Case)', 'Mean Imputation', 'Iterative Imputation','MICE Imputation']
plt.legend(labels)
plt.show()

```


    
![png](Titanic_Survival_Prediction_files/Titanic_Survival_Prediction_9_0.png)
    



```
 data = DfIterative.copy(deep=True)
 data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>22.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>38.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>26.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>35.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>53.1000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>35.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>15.300473</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.4583</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>54.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>51.8625</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.000000</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>21.0750</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>27.000000</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>11.1333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>14.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>30.0708</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```
data['Fare']= data['Fare'].replace(0,np.nan)
null_counts = data.isnull().sum()
#null
print(null_counts[null_counts > 0].sort_values(ascending=False))
#percent
print((null_counts[null_counts > 0].sort_values(ascending=False)/len(data))*100)
```

    Fare    15
    dtype: int64
    Fare    1.683502
    dtype: float64



```
%matplotlib inline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

SingleImputer = IterativeImputer(missing_values=np.nan)
DfSingle = data.copy(deep=True)
DfSingle.iloc[:,:] = SingleImputer.fit_transform(DfSingle)

MeanImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
DfMean = data.copy(deep=True)
DfMean.iloc[:, :] = MeanImputer.fit_transform(DfMean)

IterativeImputer = IterativeImputer(missing_values=np.nan, sample_posterior=True, min_value=0,random_state=0)
DfIterative = data.copy(deep=True)
DfIterative.iloc[:, :] = IterativeImputer.fit_transform(DfIterative)

data['Fare'].plot(kind='kde', c='red')
DfMean['Fare'].plot(kind='kde')
DfIterative['Fare'].plot(kind='kde', c='yellow')
DfSingle['Fare'].plot(kind='kde', c='green')
labels = ['Baseline (Initial Case)', 'Mean Imputation', 'Iterative Imputation','MICE Imputation']
plt.legend(labels)
plt.show()
```


    
![png](Titanic_Survival_Prediction_files/Titanic_Survival_Prediction_12_0.png)
    



```
data= DfIterative.copy(deep=True)
```


```
plot=data.boxplot(column='Fare',by='Pclass',figsize=(7,7))
print(plot)
Table_Fare=pd.pivot_table(data,index=['Pclass'],values=['Fare'],aggfunc=np.mean)
dictionary_Fare={'1':Table_Fare['Fare'][1],'2':Table_Fare['Fare'][2],'3':Table_Fare['Fare'][3]}

for i in range(len(data)):
  if data['Fare'][i]==0:
    if(data['Pclass'][i]==1):
      data['Fare'][i] = dictionary_Fare['1']
    elif (data['Pclass'][i]==2):
      data['Fare'][i] = dictionary_Fare['2']
    else:
      data['Fare'][i] = dictionary_Fare['3']

```

    AxesSubplot(0.1,0.15;0.8x0.75)



    
![png](Titanic_Survival_Prediction_files/Titanic_Survival_Prediction_14_1.png)
    



```
# create dummy variables for categorical features(One Hot Encoding)
data = pd.get_dummies(data,columns=['Pclass'])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Pclass_1.0</th>
      <th>Pclass_2.0</th>
      <th>Pclass_3.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>53.1000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```
f, ax = plt.subplots(figsize = [10,9])
sns.heatmap(data.corr(),linewidths = .5, annot = True, cmap = 'YlGnBu', square = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fcd01a8df28>




    
![png](Titanic_Survival_Prediction_files/Titanic_Survival_Prediction_16_1.png)
    


**3**


```
y=data['Survived']
x=data.drop(columns=['Survived'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

**KNN**


```
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print('accuracy:',knn.score(X_test, y_test))
print('recall:',recall_score(y_test, knn.predict(X_test)))
print('precision:',precision_score(y_test, knn.predict(X_test)))
print('f1_score:',f1_score(y_test, knn.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, knn.predict(X_test)))
```

    accuracy: 0.770949720670391
    recall: 0.5967741935483871
    precision: 0.6981132075471698
    f1_score: 0.6434782608695653
    confusion_matrix: 
     [[101  16]
     [ 25  37]]



```
score_knn = []
weights = ['uniform', 'distance']
for i in np.arange(1, 11, 1):
  for j in np.arange(1, 5, 1):
    for k in weights:
      knn = KNeighborsClassifier(n_neighbors=i, p=j, weights=k)
      knn.fit(X_train, y_train)
      score_knn.append([i, j, k, np.mean(cross_val_score(knn, X_train, y_train, scoring='accuracy',cv=5))])

score_knn = pd.DataFrame(score_knn)
score_knn = score_knn.sort_values(by=3, ascending=False).reset_index()
i=score_knn[0][0]
j=score_knn[1][0]
k=score_knn[2][0]
print('best parameters:','n_neighbors:',i ,'p:',j ,'weights:',k)
knn = KNeighborsClassifier(n_neighbors=i, p=j, weights=k)
knn.fit(X_train, y_train)
print('accuracy:',knn.score(X_test, y_test))
print('recall:',recall_score(y_test, knn.predict(X_test)))
print('precision:',precision_score(y_test, knn.predict(X_test)))
print('f1_score:',f1_score(y_test, knn.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, knn.predict(X_test)))

```

    best parameters: n_neighbors: 7 p: 1 weights: uniform
    accuracy: 0.7932960893854749
    recall: 0.6612903225806451
    precision: 0.7192982456140351
    f1_score: 0.689075630252101
    confusion_matrix: 
     [[101  16]
     [ 21  41]]


**Decision Tree**


```
dectree = tree.DecisionTreeClassifier(max_depth=5)
dectree.fit(X_train, y_train)
tree.plot_tree(dectree.fit(X_train, y_train))
plt.show()

print('accuracy:',dectree.score(X_test, y_test))
print('recall:',recall_score(y_test, dectree.predict(X_test)))
print('precision:',precision_score(y_test, dectree.predict(X_test)))
print('f1_score:',f1_score(y_test, dectree.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, dectree.predict(X_test)))
```


    
![png](Titanic_Survival_Prediction_files/Titanic_Survival_Prediction_23_0.png)
    


    accuracy: 0.8044692737430168
    recall: 0.5806451612903226
    precision: 0.8
    f1_score: 0.6728971962616822
    confusion_matrix: 
     [[108   9]
     [ 26  36]]



```
score_dectree = []
for i in np.arange(3, 15, 1):
  dectree = tree.DecisionTreeClassifier(max_depth=i)
  dectree.fit(X_train, y_train)
  score_dectree.append([i, np.mean(cross_val_score(dectree, X_train, y_train,scoring='accuracy',cv=5))])

score_dectree = pd.DataFrame(score_dectree)
score_dectree = score_dectree.sort_values(by=1, ascending=False).reset_index()
i=score_dectree[0][0]
print('best parameter:','max_depth:',i)

dectree = tree.DecisionTreeClassifier(max_depth=i)
dectree.fit(X_train, y_train)
tree.plot_tree(dectree.fit(X_train, y_train))
plt.show()
print('accuracy:',dectree.score(X_test, y_test))
print('recall:',recall_score(y_test, dectree.predict(X_test)))
print('precision:',precision_score(y_test, dectree.predict(X_test)))
print('f1_score:',f1_score(y_test, dectree.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, dectree.predict(X_test)))

```

    best parameter: max_depth: 3



    
![png](Titanic_Survival_Prediction_files/Titanic_Survival_Prediction_24_1.png)
    


    accuracy: 0.7653631284916201
    recall: 0.6935483870967742
    precision: 0.6515151515151515
    f1_score: 0.671875
    confusion_matrix: 
     [[94 23]
     [19 43]]


**Naive Bayes**


```
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print('accuracy:',gnb.score(X_test, y_test))
print('recall:',recall_score(y_test, gnb.predict(X_test)))
print('precision:',precision_score(y_test, gnb.predict(X_test)))
print('f1_score:',f1_score(y_test, gnb.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, gnb.predict(X_test)))

```

    accuracy: 0.7821229050279329
    recall: 0.6290322580645161
    precision: 0.7090909090909091
    f1_score: 0.6666666666666666
    confusion_matrix: 
     [[101  16]
     [ 23  39]]


**Logistic** **Regression**


```
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
C_param_range = [0.001,0.01,0.1,1,10,100]

score_log = []
for i in C_param_range:
  lr_model = LogisticRegression(penalty='l2',C=i, random_state=0)
  lr_model.fit(X_train,y_train)
  score_log.append([i, np.mean(cross_val_score(lr_model, X_train, y_train,scoring='accuracy',cv=5))])

score_log = pd.DataFrame(score_log)
score_log = score_log.sort_values(by=1, ascending=False).reset_index()
i=score_log[0][0]
print('best parameter:','C:',i)

lr_model = LogisticRegression(penalty='l2',C=i, random_state=0,)
lr_model.fit(X_train,y_train)

print('accuracy:',lr_model.score(X_test, y_test))
print('recall:',recall_score(y_test, lr_model.predict(X_test)))
print('precision:',precision_score(y_test, lr_model.predict(X_test)))
print('f1_score:',f1_score(y_test, lr_model.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, lr_model.predict(X_test)))
```

    best parameter: C: 0.1
    accuracy: 0.7541899441340782
    recall: 0.6290322580645161
    precision: 0.65
    f1_score: 0.639344262295082
    confusion_matrix: 
     [[96 21]
     [23 39]]


**Bagging**


```
bag = BaggingClassifier(n_estimators=100)
bag.fit(X_train, y_train)

print('accuracy:',bag.score(X_test, y_test))
print('recall:',recall_score(y_test, bag.predict(X_test)))
print('precision:',precision_score(y_test, bag.predict(X_test)))
print('f1_score:',f1_score(y_test, bag.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, bag.predict(X_test)))
```

    accuracy: 0.7821229050279329
    recall: 0.6290322580645161
    precision: 0.7090909090909091
    f1_score: 0.6666666666666666
    confusion_matrix: 
     [[101  16]
     [ 23  39]]



```
score_bag = []
for i in np.arange(5, 110, 5):
  bag = BaggingClassifier(n_estimators=i)
  bag.fit(X_train, y_train)
  score_bag.append([i, np.mean(cross_val_score(bag , X_train, y_train,scoring='accuracy',cv=5))])

score_bag = pd.DataFrame(score_bag)
score_bag = score_bag.sort_values(by=1, ascending=False).reset_index()
i=score_bag[0][0]
print('best parameter:','n_estimators:',i)
bag = BaggingClassifier(n_estimators=i)
bag.fit(X_train, y_train)

print('accuracy:',bag.score(X_test, y_test))
print('recall:',recall_score(y_test, bag.predict(X_test)))
print('precision:',precision_score(y_test, bag.predict(X_test)))
print('f1_score:',f1_score(y_test, bag.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, bag.predict(X_test)))
```

    best parameter: n_estimators: 15
    accuracy: 0.7653631284916201
    recall: 0.6290322580645161
    precision: 0.6724137931034483
    f1_score: 0.6499999999999999
    confusion_matrix: 
     [[98 19]
     [23 39]]


**Random Forest**


```
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print('accuracy:',rf.score(X_test, y_test))
print('recall:',recall_score(y_test, rf.predict(X_test)))
print('precision:',precision_score(y_test, rf.predict(X_test)))
print('f1_score:',f1_score(y_test, rf.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, rf.predict(X_test)))
```

    accuracy: 0.7877094972067039
    recall: 0.6612903225806451
    precision: 0.7068965517241379
    f1_score: 0.6833333333333333
    confusion_matrix: 
     [[100  17]
     [ 21  41]]



```
score_rf = []
max_features = ['auto', 'sqrt']

for i in np.arange(100, 501, 100):
  for j in np.arange(5, 21 , 5):
    for k in max_features:
      rf = RandomForestClassifier()
      rf.fit(X_train, y_train)
      score_rf.append([i, j, k, np.mean(cross_val_score(rf, X_train, y_train, scoring='accuracy',cv=5))])

score_rf = pd.DataFrame(score_rf)
score_rf = score_rf.sort_values(by=3, ascending=False).reset_index()
i=score_rf[0][0]
j=score_rf[1][0]
k=score_rf[2][0]
print('best parameters:','n_estimators:',i ,'max_depth:',j ,'max_features:',k)

rf = RandomForestClassifier(n_estimators=i, max_depth=j, max_features=k)
rf.fit(X_train,y_train)

print('accuracy:',rf.score(X_test, y_test))
print('recall:',recall_score(y_test, rf.predict(X_test)))
print('precision:',precision_score(y_test, rf.predict(X_test)))
print('f1_score:',f1_score(y_test, rf.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, rf.predict(X_test)))

```

    best parameters: n_estimators: 100 max_depth: 15 max_features: auto
    accuracy: 0.7877094972067039
    recall: 0.6451612903225806
    precision: 0.7142857142857143
    f1_score: 0.6779661016949152
    confusion_matrix: 
     [[101  16]
     [ 22  40]]


**Support Vector Machines**


```
parameters_svm = {'C':[0.9,0.01],'kernel':['rbf','linear'], 'gamma':[0,0.1,'auto'], 'probability':[True,False],'degree':[3,4,10]}
clf_svm = SVC()

def grid(model,parameters):
    grid = GridSearchCV(estimator = model, param_grid = parameters, cv = 10, scoring = 'accuracy')
    grid.fit(X_train,y_train)
    return grid.best_score_, grid.best_estimator_.get_params()

best_score_svm, best_params_svm = grid(clf_svm, parameters_svm)
print(best_score_svm)
print(best_params_svm)

```

    0.8314945226917058
    {'C': 0.9, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}



```
clf_svm = SVC(C=0.9,kernel='rbf',gamma='auto')
clf_svm.fit(X_train,y_train)

print('accuracy:',clf_svm.score(X_test, y_test))
print('recall:',recall_score(y_test, clf_svm.predict(X_test)))
print('precision:',precision_score(y_test, clf_svm.predict(X_test)))
print('f1_score:',f1_score(y_test, clf_svm.predict(X_test)))
print('confusion_matrix:','\n',confusion_matrix(y_test, clf_svm.predict(X_test)))

```

    accuracy: 0.7653631284916201
    recall: 0.5645161290322581
    precision: 0.7
    f1_score: 0.625
    confusion_matrix: 
     [[102  15]
     [ 27  35]]

