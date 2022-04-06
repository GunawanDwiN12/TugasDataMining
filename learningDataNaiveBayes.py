import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn .model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
dataset = pd.read_csv('Sosial Network.csv')
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test= train_test_split(x, y , test_size=0.25, random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.fit_transform(x_test)
print(x_train)
print(x_test)


classifier = GaussianNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
cm =confusion_matrix(y_test,y_pred)
print(cm)
x_set, y_set = x_train, y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop =x_set[:,0].max()+1,step=0.01),
                    np.arange(start = x_set[:,1].min()-1,stop =x_set[:,0].max()+1,step=0.01))


plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('blue','grey')))      
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate (np.unique(y_test)):
    plt.scatter(x_set[y_set==j ,0], x_set[y_set==j,1], c = ListedColormap(('red','yellow'))(i),label=j)
plt.title('KLASIFIKASI GAJI KARYAWAN')
plt.xlabel('Umur')
plt.ylabel('Estimasi gaji')
plt.legend()
plt.show()