# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 18:21:12 2019

@author: 2003065
"""

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing

# load dataset
df = pd.read_csv("D:/Final/Modeling/features_50_v2_201912241045/features_50_v2_201912241045.csv", header=0)
df.head()
df.dtypes

df['vehicle_age_bin'].nunique(), df['vehicle_age_bin'].unique()
df['vehicle_age_bin'].value_counts()
#df['vehicle_age_bin_mod'].value_counts()

#Combining Bins for Vehicle Age Bin
df.loc[(df['vehicle_age_bin'] == 'Loyal(GT 10Y)') |
       (df['vehicle_age_bin'] == 'Sapphire(8Y-10Y)')|
       (df['vehicle_age_bin'] == 'Platinum(4Y-5Y)')|
       (df['vehicle_age_bin'] == 'Diamond(6Y-7Y)'), 'vehicle_age_bin'] = 'Others' 

#Dropping Missing Values
df.shape
df=df.dropna()
df.shape

#Target Variable Encoding
encoder = preprocessing.LabelEncoder()
df["labels"].value_counts()
df["labels"] = encoder.fit_transform(df["labels"])#.fillna('Nan'))
df["labels"].value_counts()

ds_cat = df.select_dtypes(include = 'object').copy()
ds_num = df.select_dtypes(exclude = 'object').copy()
ds_cat.head()
ds_num.head()


#Label Encoding for all categorical variable
for i in ds_cat.columns:
    ds_cat[i] = encoder.fit_transform(ds_cat[i])
    
#Concatenating again #cat & #num
df_new = pd.concat([ds_num, ds_cat], axis=1)

    
'''
#Dummy Coding Categorical Variables
df_dummies = pd.get_dummies(ds_cat)
df_new = pd.concat([ds_num, df_dummies], axis=1)
df_new.head()
X=df_new[['avg_odo_rd', 'dis_vin_ct', 'avg_veh_age', 'hat_ct', 'sed_ct', 'suv_ct',
       'die_ct', 'gas_ct', 'l_gas_ct', 'pet_ct', 'u_gas_ct', 'fre_an_ct',
       'fre_as_ct', 'fre_bi_ct', 'fre_on_ct', 'type_fes_ct', 'type_fla_ct',
       'type_oth_ct', 'type_sea_ct', 'type_spe_ct', 'max_con_cd_B',
       'max_con_cd_C', 'max_con_cd_E', 'max_con_cd_F', 'max_con_cd_G',
       'max_con_cd_I', 'max_sub_type_Festival', 'max_sub_type_Flagship',
       'max_sub_type_Seasonal', 'max_sub_type_Special Day',
       'max_frequency_Annual', 'max_frequency_Bi-Annual', 'car_own_ty_Multi',
       'car_own_ty_Single', 'vehicle_age_bin_Diamond(6Y-7Y)',
       'vehicle_age_bin_Gold(0-3Y)', 'vehicle_age_bin_Loyal(GT 10Y)',
       'vehicle_age_bin_Platinum(4Y-5Y)', 'vehicle_age_bin_Sapphire(8Y-10Y)',
       'max_fuel_cd_Diesel', 'max_fuel_cd_Gasoline',
       'max_fuel_cd_Leaded Gasoline', 'max_fuel_cd_Petrol',
       'max_fuel_cd_Unleaded Gasoline', 'max_veh_type_Hatchback',
       'max_veh_type_SUV', 'max_veh_type_Sedan']]
'''

#Create X Y objects
X=df_new.loc[:,df_new.columns!='labels']
y=df_new['labels']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=0)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))


#Logistic Regression
logreg_clf = LogisticRegression()
log = logreg_clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = log.predict(X_test)
# Model Evaluation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))
