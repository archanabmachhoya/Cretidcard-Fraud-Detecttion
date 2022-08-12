#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#load dataset


# In[4]:


data = pd.read_csv(r"D:\Archana1\DataScience_GTU\My_Project\Credit card fraud detection_Cloud x Lab_project\creditcard.csv")


# In[5]:


data.head(10)


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


#checking for nulls


# In[9]:


data.isnull()


# In[10]:


data.isnull().sum()


# In[11]:


#explronig the class cell


# In[12]:


X = data.loc[:, data.columns != 'Class']


# In[13]:


y = data.loc[:, data.columns == 'Class']


# In[14]:


X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']


# In[15]:


print(X)


# In[16]:


print(y)


# In[17]:


print(data['Class'].value_counts())


# In[18]:


print('Valid Transactions: ', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')


# In[19]:


len(data)


# In[20]:


print('Fraudulent Transactions: ', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')


# In[21]:


print(data['Class'].value_counts())


# In[22]:


#Visualizing the class Imbalance


# In[23]:


colors = ['blue','red']

sns.countplot(x='Class', data=data, palette=colors)


# In[24]:


#Splitting the Data


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[26]:


print("Transactions in X_train dataset: ", X_train.shape)


# In[27]:


print("Transactions in X_test dataset: ", X_test.shape)


# In[28]:


print("Transactions in y_train dataset: ", y_train.shape)


# In[29]:


print("Transactions in y_testdataset: ", y_test.shape)


# In[30]:


#Feature Scaling


# In[31]:


from sklearn.preprocessing import StandardScaler
scaler_amount = StandardScaler()
scaler_time = StandardScaler()

X_train['normAmount'] = scaler_amount.fit_transform(X_train['Amount'].values.reshape(-1, 1))
X_test['normAmount'] = scaler_amount.transform(X_test['Amount'].values.reshape(-1, 1))

X_train['normTime'] = scaler_time.fit_transform(X_train['Time'].values.reshape(-1, 1))
X_test['normTime'] = scaler_time.transform(X_test['Time'].values.reshape(-1, 1))

X_train = X_train.drop(['Time', 'Amount'], axis=1)
X_test = X_test.drop(['Time', 'Amount'], axis=1)

X_train.head()


# In[32]:


#Applying SMOTE technique


# In[33]:


pip install imblearn


# In[34]:


from imblearn.over_sampling import SMOTE
print("Before over-sampling\n", y_train['Class'].value_counts())


# In[35]:


sm = SMOTE()


# In[36]:


X_train_res, y_train_res = sm.fit_resample(X_train, y_train['Class'])
print("After over-sampling:\n", y_train_res.value_counts())


# In[37]:


#Training the Classification Algorithm


# In[38]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve

parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lr = LogisticRegression()

clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)

k = clf.fit(X_train_res, y_train_res)

print(k.best_params_)


# In[39]:


#Get Confusion matrix and Recall


# In[40]:


lr_gridcv_best = clf.best_estimator_

y_test_pre = lr_gridcv_best.predict(X_test)

cnf_matrix_test = confusion_matrix(y_test, y_test_pre)

print(cnf_matrix_test)
print("Recall metric in the test dataset:", (cnf_matrix_test[1,1]/(cnf_matrix_test[1,0]+cnf_matrix_test[1,1] )))


y_train_pre = lr_gridcv_best.predict(X_train_res)

cnf_matrix_train = confusion_matrix(y_train_res, y_train_pre)

print(cnf_matrix_test)
print("Recall metric in the train dataset:", (cnf_matrix_train[1,1]/(cnf_matrix_train[1,0]+cnf_matrix_train[1,1] )))


# In[41]:


#*******************************************
#another way to predict model


# In[42]:


from sklearn.linear_model import LogisticRegression


# In[43]:


Lr = LogisticRegression()


# In[44]:


Lr.fit (X_train_res, y_train_res)


# In[45]:


Lr.predict(X_test)


# In[46]:


Lr.score(X_test, y_test)


# In[47]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,Lr.predict(X_test))


# In[48]:


from sklearn.metrics import classification_report
print(classification_report(y_test, Lr.predict(X_test)))


# In[49]:


#***********************************************************************************


# In[50]:


#visualize the Confusion Matrix


# In[51]:


from sklearn.metrics import plot_confusion_matrix

class_names = ['Not Fraud', 'Fraud']

plot_confusion_matrix(k, X_test, y_test,  values_format = '.5g', display_labels=class_names) 
plt.title("Test data Confusion Matrix")
plt.show()

plot_confusion_matrix(k, X_train_res, y_train_res,  values_format = '.5g', display_labels=class_names)
plt.title("Oversampled Train data Confusion Matrix")
plt.show()


# In[52]:


#ROC-AUC Curve (Receiver Operator Characteristic,Area Under the Curve )


# In[53]:


y_k =  k.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_k)

roc_auc = auc(fpr,tpr)

print("ROC-AUC:", roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[54]:


#predict on new data


# In[55]:


data1 = pd.read_csv(r"D:\Archana1\DataScience_GTU\My_Project\Credit card fraud detection_Cloud x Lab_project\New data_creditcard .csv")


# In[56]:


data1.shape


# In[57]:


a=Lr.predict(data1)
print(a)


# In[58]:


Lr.score(data1,a)


# In[59]:


#use 3rd type of data with max NAN values in using mean values to fill the data


# In[60]:


import pandas as pd


# In[61]:


data2 = pd.read_csv(r"D:\Archana1\DataScience_GTU\My_Project\Credit card fraud detection_Cloud x Lab_project\new data _1_creditcard.csv")


# In[62]:


data2


# In[63]:


import numpy as np
from sklearn.impute import SimpleImputer


# In[64]:


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
b=imputer.fit(data2)


# In[65]:


data2.shape


# In[66]:


imputer.statistics_


# In[67]:


c=imputer.transform(data2)
c


# In[68]:


y= pd.DataFrame(c)
y


# In[69]:


y.describe()


# In[70]:


c.shape


# In[71]:


c1=Lr.predict(c)
c1


# In[72]:


Lr.score(c,c1)


# In[ ]:




