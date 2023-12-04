#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
# Register converters to avoid warnings
pd.plotting.register_matplotlib_converters()
plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)
import warnings
warnings.filterwarnings("ignore")
# 讀檔
df = pd.read_csv('D:/Anna/Anna/zhuanti/data.csv')


# In[2]:


# 原本為2021-11-01-00，現將其拆解並新增欄位存取資料
df['month'] = df['time'].str.split('-', expand = True)[1]
df['month'] = df['month'].astype(int)
df['date'] = df['time'].str.split('-', expand = True)[2]
df['date'] = df['date'].astype(int)
df['hour'] = df['time'].str.split('-', expand = True)[3]
df['hour'] = df['hour'].astype(int)


# In[3]:


#資料整理
df['weekday'] = df['date']
df['type'] = df['date']
df['bool_weekday'] = df['hour']

for i in range(len(df)):
    if df['month'][i] == 11:
        if df['date'][i] % 7 == 1:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 2:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 3:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 4:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 5:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 6:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'
    elif df['month'][i] == 12:
        if df['date'][i] % 7 == 6:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 0:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 1:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 2:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 3:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 4:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'
    
    elif df['month'][i] == 1:
        if df['date'][i] % 7 == 3:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 4:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 5:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 6:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 0:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 1:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'
    
    
    elif df['month'][i] == 2:
        if df['date'][i] % 7 == 0:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 1:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 2:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 3:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 4:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 5:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'
# new data            
    elif df['month'][i] == 3:
        if df['date'][i] % 7 == 0:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 1:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 2:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 3:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 4:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 5:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'
            
    elif df['month'][i] == 4:
        if df['date'][i] % 7 == 4:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 5:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 6:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 0:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 1:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 2:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'
            
    elif df['month'][i] == 5:
        if df['date'][i] % 7 == 2:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 3:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 4:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 5:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 6:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 0:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'
            
    elif df['month'][i] == 6:
        if df['date'][i] % 7 == 6:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 0:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 1:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 2:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 3:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 4:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'
            
    elif df['month'][i] == 7:
        if df['date'][i] % 7 == 4:
            df['weekday'][i] = 'Mon'
            df['type'][i] = 'weekday'
        elif df['date'][i] % 7 == 5:
            df['weekday'][i] = 'Tue'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 6:
            df['weekday'][i] = 'Wed'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 0:
            df['weekday'][i] = 'Thu'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 1:
            df['weekday'][i] = 'Fri'
            df['type'][i] = 'weekday'
        elif df['date'][i]  % 7 == 2:
            df['weekday'][i] = 'Sat'
            df['type'][i] = 'weekend'
        else:
            df['weekday'][i] = 'Sun'
            df['type'][i] = 'weekend'

for i in range(len(df)):
    if df['type'][i] == 'weekday':
        df['bool_weekday'][i] = 1
    else:
        df['bool_weekday'][i] = 0


# In[4]:


df['volume'] = df['stateRun'] + df['private']
df.set_index('time', inplace = True)


# In[5]:


df


# In[6]:


df = df.dropna()


# In[13]:


# reference : arima時間序列模型python應用-銅價格預測 農二代

# 新增外在變數，每小時太陽光照時長、風速與是否為平假日
exog = sm.add_constant(df[['hour','bool_weekday']])
arima_data = df['volume']


# In[14]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
exog =stdsc.fit_transform(exog)


# In[15]:


# Fit the model
# 由下表可得到運算公式，可看到日照時間*-17.8224，風速*1.7994，平日*-40.2452，特殊日-50.5804，溫度*1.9907
# 因有季節性變化，因此加設seasonal_order，每24一個循環
# (3,0,2)、(1,0,1,24)由前述公式而得
mod = sm.tsa.statespace.SARIMAX(arima_data, exog, order=(2,0,1), seasonal_order=(1,0,1,24), simple_differencing=True)
res = mod.fit(disp=False)
print(res.summary())


# In[16]:


endog = arima_data
mod = sm.tsa.statespace.SARIMAX(endog, exog=exog, order=(2,0,1), seasonal_order=(1,0,1,24))
res = mod.filter(res.params)


# In[17]:


# In-sample one-step-ahead predictions
predict = res.get_prediction()
predict_ci = predict.conf_int()


# In[18]:


predict


# In[21]:


ppm = predict.predicted_mean
ppm = ppm.reset_index()
ppm['time'] = ""
#ppm.drop(labels=['index'], axis=1, inplace=True)
arima_new = arima_data
arima_new = arima_new.reset_index()
p = predict_ci
p = p.reset_index()
#p.drop(labels=['index'], axis=1, inplace=True)
p['time'] = ""


# In[23]:


p


# In[22]:


ppm


# In[24]:


for i in range(len(ppm)):
    ppm['time'][i] =arima_new['time'][i]
    p['time'][i] = arima_new['time'][i]


# In[25]:


ppm = ppm.set_index('time')
p = p.set_index('time')
arima_new = arima_new.set_index('time')


# In[26]:


p


# In[27]:


# Graph
fig, ax = plt.subplots(figsize=(20,8))
npre = 4
ax.set(title='Predict',xlabel = 'time', ylabel='volume')

# Plot data points
arima_new.plot(ax=ax, style='-', label='Observed')
# Plot predictions
ppm['predicted_mean'].plot(ax=ax, style='r--', label='One-step-ahead forecast')
#ax.fill_between(p.index, p.iloc[:,0], p.iloc[:,1], color='r', alpha=0.1)
plt.ylim([0,700])
legend = ax.legend(["Observed", "One-step-ahead forecast",'Dynamic forecast'],loc='upper right')


# In[28]:


#計算error
predict_error = ppm['predicted_mean'] - arima_new['volume']
predict_error= predict_error.reset_index()
predict_error['time'] = ""

ci_new = p
ci_new.iloc[:,0] -= arima_new['volume']
ci_new.iloc[:,1] -= arima_new['volume']

for i in range(len(ppm)):
    predict_error['time'][i] = df.reset_index()['time'][i]
    
predict_error = predict_error.set_index('time')


# In[29]:


# Prediction error

# Graph
fig, ax = plt.subplots(figsize=(20,8))
npre = 4
ax.set(title='Forecast error', xlabel='Date', ylabel='Forecast - Actual')

# In-sample one-step-ahead predictions and 95% confidence intervals

predict_error.plot(ax=ax, style = '-',label='One-step-ahead forecast')

ax.fill_between(ci_new.index, ci_new.iloc[:,0], ci_new.iloc[:,1], alpha=0.1)



legend = ax.legend(loc='lower left');
legend.get_frame().set_facecolor('w')
plt.ylim([-500,500])


# In[30]:


squaredError = []
absError = []
for i in predict_error[0]:
    squaredError.append(i * i)
    absError.append(abs(i))


# In[31]:


print("Square Error: ",squaredError)
print("Absolute Value of Error: ",absError)


# In[32]:


print("MSE = ",sum(squaredError) / len(squaredError))#均方誤差MSE


# In[33]:


from math import sqrt
print("RMSE = ",sqrt(sum(squaredError) / len(squaredError)))#均方根誤差RMSE
print("MAE = ",sum(absError) / len(absError))#平均絕對誤差MAE


# In[34]:


df = df.reset_index()


# # 將誤差存起來並加入BPN模型中

# In[125]:


df['error'] = df['month']
time = 0
for i in predict_error[0]:
    df['error'][time] = i
    time += 1


# In[126]:


df['datatype'] = df['bool_weekday']
for i in range(len(df)):
    if df['type'][i] == 'weekday':
        df['bool_weekday'][i] = 1
    else:
        df['bool_weekday'][i] = 0
    if df['volume'][i] > 320:
        df['datatype'][i] = 2
    elif df['volume'][i] > 100:
        df['datatype'][i] = 1
    else:
        df['datatype'][i] = 0
# 分法 : 對照總資料的describe，發現前25% 94 後75% 300，化整來區分


# In[127]:


df


# In[128]:


# reference : Python Machine Learning, 3rd Ed.
'''
The MIT License (MIT)

Copyright (c) 2019 SEBASTIAN RASCHKA (mail@sebastianraschka.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import sys

class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training examples per minibatch.
    seed : int (default: None)
        Random seed for initializing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_examples]
            Target values.
        n_classes : int
            Number of classes

        Returns
        -----------
        onehot : array, shape = (n_examples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        idx = 0
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_examples, n_features] dot [n_features, n_hidden]
        # -> [n_examples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # [n_examples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_examples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_examples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_examples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        
        return cost

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_examples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_examples]
            Predicted class labels.

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_examples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_examples]
            Target class labels.
        X_valid : array, shape = [n_examples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_examples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                
                
                z_h, a_h, z_out, a_out = self._forward(X_train)

                ##################
                # Backpropagation
                ##################

                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc

                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # [n_features, n_examples] dot [n_examples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train.T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # [n_hidden, n_examples] dot [n_examples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            
            z_h, a_h, z_out, a_out = self._forward(X_train)
            
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self


# In[198]:


X = pd.DataFrame(df['hour'])
X = pd.concat([X, pd.DataFrame(df['bool_weekday'])],axis=1)
X = pd.concat([X, pd.DataFrame(df['sunshineHour'])],axis=1)
X = pd.concat([X, pd.DataFrame(df['windSpeed'])],axis=1)
X = pd.concat([X, pd.DataFrame(df['special'])],axis=1)
X = pd.concat([X, pd.DataFrame(df['temperature'])],axis=1)
X = pd.concat([X, pd.DataFrame(df['error'])],axis=1)

y = df.datatype
print('Class labels:', np.unique(y))


# In[199]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1, stratify = y)


# In[200]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train =stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)


# In[201]:


n_epochs = 200

nn = NeuralNetMLP(n_hidden=30, 
                  l2=0.01, 
                  epochs=n_epochs, 
                  eta=0.0001,
                  minibatch_size=100, 
                  shuffle=True,
                  seed=1)

nn.fit(X_train = X_train[:2000], 
       y_train = y_train[:2000],
       X_valid=X_train[2000:],
       y_valid=y_train[2000:])


# In[202]:


import matplotlib.pyplot as plt
plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()


# In[203]:


plt.plot(range(nn.epochs), nn.eval_['train_acc'],
 label = 'training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'],
 label = 'validation', linestyle = '--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc = 'lower right')
plt.show()


# In[204]:


y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
print('Training accuracy: %.2f%%' % (acc *100))


# In[205]:


y_test_pred = nn.predict(X_test)


# In[206]:


y_test = y_test.reset_index()
del y_test['index']


# # 若僅考慮預測出的高與低是否為正確預測

# In[207]:


time = 0
acc_2 = 0
for i in range(len(y_test_pred)):
    if y_test_pred[i] == 0:
        time += 1
        if y_test['datatype'][i] == y_test_pred[i]:
            acc_2+= 1
    if y_test_pred[i] == 2:
        time += 1
        if y_test['datatype'][i] == y_test_pred[i]:
            acc_2+= 1  
acc_2 = acc_2 / time
print('Training accuracy: %.2f%%' % (acc_2 *100))


# In[ ]:




