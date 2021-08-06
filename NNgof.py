import tensorflow as tf
import random
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

training1 = np.random.multinomial(20, [1/6.]*20, size=100)

training3 = np.random.multinomial(25, [1/6.]*6, size=100)
training4 = np.random.multinomial(15, [1/6.]*6, size=100)

class_names=["H0", "H1"]
x = training1
y = np.zeros(len(training1))
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