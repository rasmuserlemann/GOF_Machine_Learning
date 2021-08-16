import tensorflow as tf
import random
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import seaborn as sns

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

trainx = []
trainy = []

CL1p = c(0, 0.013906448, 0.039190898, 0.041297935, 0.004214075, 0.015170670, 0.025705858, 0.044669195, 0.008849558, 0.432364096, 0.240202276, 0.006742520, 0.034134008, 0.034134008, 0.025284450, 0.013063633, 0.021070375, 0)
CL2p = c(0, 0, 0, 0, 0.031250000, 0, 0, 0, 0.058035714, 0, 0.750000000, 0.031250000, 0.022321429, 0.084821429, 0.008928571, 0.008928571, 0.004464286, 0)
CL3p = c(0, 0, 0.032258065, 0.064516129, 0, 0, 0.032258065, 0, 0.361290323, 0.135483871, 0, 0.006451613, 0.012903226, 0.316129032, 0.012903226, 0.006451613, 0.019354839, 0)
CL4p = c(0.205655527, 0, 0, 0, 0.015424165, 0, 0, 0, 0.023136247, 0.205655527, 0.318766067, 0.007712082, 0, 0.017994859, 0, 0, 0, 0.205655527)
CL5p = c(0, 0, 0, 0, 0.3333333, 0, 0, 0, 0, 0.3333333, 0, 0.3333333, 0, 0, 0, 0, 0, 0)
CL6p = c(0, 0.01470588, 0.11764706, 0.45588235, 0, 0.04411765, 0.10294118, 0.05882353, 0, 0, 0.08823529, 0, 0.01470588, 0.02941176, 0, 0.01470588, 0.05882353, 0)

trainsize = 1000
for ind in range(trainsize):
    trainx.append(np.concatenate((np.random.multinomial(20, [1/6.]*6, size=1), np.random.multinomial(20, [1/6.]*6, size=1)), axis=1)[0])
    trainy.append(0)
    trainx.append(np.concatenate((np.random.multinomial(20, [1/2, 1/8, 1/16, 2/16, 1/16, 2/16], size=1), np.random.multinomial(20, [1/6.]*6, size=1)), axis=1)[0])
    trainy.append(1)



class_names=["H0", "H1"]
x = trainx
y = trainy
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
tick_marks = range(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
