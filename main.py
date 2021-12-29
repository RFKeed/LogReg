# **IMPORTS**


from numpy.matrixlib.defmatrix import matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

from LogReg import LogisticRegression

# **DATA**
print('DATA')
X_train = np.random.randn(30, 3)
y_train = np.array(list(map(lambda x: np.sum(x) > 0, X_train))).reshape(-1, 1).astype(np.int32)
print(X_train, y_train)
# **TRAIN OUR MODEL AND SEE LOSS**
print('TRAIN OUR MODEL AND SEE LOSS')
log_reg = LogisticRegression()
losses = log_reg.fit(X_train, y_train, iters=1000)

plt.figure(figsize=(10, 8))
plt.plot(losses, label='loss')
plt.xlim(0, len(losses))
plt.grid()
plt.legend()
plt.show()

# Now Let's see how real model works
print("Now Let's see how real model works")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
print(x, y)

plt.figure(figsize=(10, 8))
plt.plot(x, label='x')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(y, label='y')
plt.grid()
plt.legend()
plt.show()

model = LogisticRegression(solver='liblinear', random_state=0)

LogisticRegression(
    C=1.0, 
    class_weight=None, 
    dual=False, 
    fit_intercept=True,
    intercept_scaling=1, 
    l1_ratio=None, 
    max_iter=100,
    multi_class='warn', 
    n_jobs=None, 
    penalty='l2',
    random_state=0, 
    solver='liblinear', 
    tol=0.0001, 
    verbose=0,
    warm_start=False
)

model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)

model.predict_proba(x)
model.predict(x)

print(log_reg.accuracy(x, y))

cm = confusion_matrix(y, model.predict(x))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()