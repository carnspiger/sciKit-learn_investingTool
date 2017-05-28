import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm   #support vector machine, creates vectors to separate groups that you designated

digits = datasets.load_digits()

print(digits.data)
print(digits.target)
print(digits.images[0])

#setup classifier
clf = svm.SVC(gamma=0.001, C=100)   #can automatically pick best version of gamma using machine learning

print(len(digits.data))   #1797 is default size of digit dataset

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)   #overfitting will circle around the outside datapoints on each cluster  --> bad alg

print('Prediction:',clf.predict(digits.data[-2]))

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")  #zooms in on plot
plt.show()