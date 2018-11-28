
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.utils.np_utils import to_categorical
import time

def loadData(filename):
    rawData = pd.read_csv(filename)
    closePrice = np.array(rawData.loc[:, [' CLOSE']])
    return   closePrice[:,0]

def create_dataset(closePrice,look_back,f_horizon):    
    dataX = []
    dataY = []
    dataL=[]
    for i in range(0,len(closePrice)-look_back-f_horizon,1):
        a = closePrice[i:i+look_back]
        b = closePrice[i+look_back:i+look_back + f_horizon]
        dataX.append(a.tolist())
        dataY.append(b.tolist())
    return dataX, dataY

def create_dataset_discrete(closePrice,look_back):    
    dataX = []
    dataY = []
    dataY_raw = []
        
    for i in range(0,len(closePrice)-look_back-1):
        a = closePrice[i:i+look_back]
        end=closePrice[i+look_back-1]
        b = (closePrice[i+look_back] -end)/ (end) * 100
        labels = 0
        if b > 0.25:
            labels = 1
        elif b < -0.25:
            labels = 2
        dataX.append(a.tolist())
        dataY.append(labels)
        dataY_raw.append(b)
        
    return dataX, dataY, dataY_raw


def normalize_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data)
    norm_data=min_max_scaler.transform(data)
    return norm_data,min_max_scaler


def exp_moving_avg(data):
    ema = 0.0
    gamma = 0.1
    for ti in range(len(data)):
        ema = gamma*data[ti] + (1-gamma)*ema
        data[ti] = ema
    return data


def visualize(data):
    plt.plot(data)
    plt.show()
    
def hist_visualize(dataY, x_label_name,y_label_name, figure_title):
    n, bins, patches = plt.hist(x=dataY, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.title(figure_title)
    plt.text(23, 45, r'$\mu=15, b=3$')
    plt.ylim()
    return plt.show()


def train_test_split(dataX,dataY,percent_split):
    row = int(percent_split * dataX.shape[0])
    trainX = dataX[:row, :]
    testX=dataX[row:,:]
    trainY=dataY[:row,:]
    testY=dataY[row:,:]
    return trainX,trainY,testX,testY


filename = '../input/stock_data.txt' 
look_back = 8  
percent_train_data = .7 

closePrice= loadData(filename)
norm_closePrice,scaler=normalize_data(np.array(closePrice).reshape(-1,1))
ema=exp_moving_avg(norm_closePrice)
dataX,dataY,dataYraw=create_dataset_discrete(ema,look_back)


X=np.array(dataX).reshape(-1,1,look_back)
Y_dat=[]
for i in range(len(dataY)):
    Y_dat.append(to_categorical(dataY[i],3).reshape(1,3))
Y=np.array(Y_dat)

print(X.shape)
print(Y.shape)

X_train,Y_train,X_test,Y_test=train_test_split(X,Y,0.9)
print(X_train.shape)
print(Y_train.shape)


def model(look_back, mode):
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(1, look_back), merge_mode=mode))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def train_model(model):
    hist=model.fit(X_train,Y_train,batch_size=64,nb_epoch=250)
    return hist

bi_sum=model(look_back,'sum')
sum_hist=train_model(bi_sum)
bi_mul=model(look_back, 'mul')
mul_hist=train_model(bi_mul)
bi_ave=model(look_back,'ave')
ave_hist=train_model(bi_ave)
bi_concat=model(look_back,'concat')
con_hist=train_model(bi_concat)

results=pd.DataFrame()
results['sum']=sum_hist.history['loss']
results['mul']=mul_hist.history['loss']
results['ave']=ave_hist.history['loss']
results['con']=con_hist.history['loss']
results.plot()
plt.show()

model=model(look_back,'sum')
model.summary()

vals=model.fit(X_train,Y_train,batch_size=64,nb_epoch=1000)

visualize(vals.history['loss'])

preds=model.predict_classes(X_test)
orig=np.array(dataY[6354:]).reshape(-1,1)
accuracy=(preds==orig).sum()/preds.shape[0]
print(accuracy)

hist_visualize(orig, 'Class Labels','frequency', 'class_distribution')

