import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
#réseau mono-couche
from sklearn.linear_model import SGDClassifier
# Pour ignorer les messages d'avertissement
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.neural_network import MLPClassifier

data3 = np.loadtxt("dataset3.txt")
X = data3[:,:2]
y = data3[:,2]

for i in range(len(y)):
    temps = y[i]
    y[i]= temps

y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state=1)

colors = np.array([x for x in "rgbcmyk"])
# scatter = plt.scatter(X[:,0],X[:,1], c=y, s=10)
# plt.legend(scatter.legend_elements(),bbox_to_anchor=(1.05, 1.0), loc='best',title="donnée d'apprentissage")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.show()

#learning rate
A = 0.6
#epoch
B = 30
#hidden layer size
C = 3

### Dataset 1

# acc_train = []
# for rep in range(5):
#     clf1=SGDClassifier(loss='perceptron', eta0=A, max_iter=B, learning_rate='constant')
#     clf1.fit(X_train,y_train)

#     #score sur la base d'apprentissage
#     print(f"accuracy on training set for run {rep+1}: {clf1.score(X_train,y_train)}" )
#     acc_train.append(clf1.score(X_train,y_train))

# print('mean accuracy = ', np.mean(acc_train))
# print('std accuracy = ', np.std(acc_train))

# # Créer une grille
# x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
# y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
# x_h = (x_max - x_min)/50
# y_h = (y_max - y_min)/50
# xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
# np.arange(y_min, y_max, y_h))
# Y = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
# Y = Y.reshape(xx.shape)
# plt.grid()

# #afficher les frontières/données d'apprentissage
# plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X_train[:,0], X_train[:,1:2], color=colors[y_train].tolist(), s=10)
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.title("Frontières de décision")
# plt.show()

### Dataset 2

# clf2  = MLPClassifier(hidden_layer_sizes=C, activation='tanh', solver='adam', learning_rate_init=A, max_iter=B, learning_rate='adaptive', shuffle=True, batch_size=len(y_train))

# clf2.fit(X_train, y_train)

# print('best loss = ', clf2.best_loss_)

# plt.plot(clf2.loss_curve_)
# plt.grid()
# plt.xlabel('Epoque')
# plt.ylabel('Loss')
# plt.show()

# # Créer une grille
# x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
# y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
# x_h = (x_max - x_min)/50
# y_h = (y_max - y_min)/50
# xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
# np.arange(y_min, y_max, y_h))
# Y = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
# Y = Y.reshape(xx.shape)
# plt.grid()

# #afficher les frontières/données d'apprentissage
# plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X_train[:,0], X_train[:,1:2], color=colors[y_train].tolist(), s=10)
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.title("Frontières de décision")
# plt.show()

### Dataset 3

acc_train = []
for rep in range(1,6):
    clf3=MLPClassifier(hidden_layer_sizes=C, activation='tanh', solver='adam', learning_rate_init=A, max_iter=B, learning_rate='adaptive', shuffle=True, batch_size=len(y_train))
    clf3.fit(X_train,y_train)

    #score sur la base d'apprentissage
    print(f"accuracy on training set for run {rep+1}: {clf3.score(X_train,y_train)}" )
    acc_train.append(clf3.score(X_train,y_train))

print('mean accuracy = ', np.mean(acc_train))
print('std accuracy = ', np.std(acc_train))

# Créer une grille
x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
np.arange(y_min, y_max, y_h))
Y = clf3.predict(np.c_[xx.ravel(), yy.ravel()])
Y = Y.reshape(xx.shape)
plt.grid()

#afficher les frontières/données d'apprentissage
plt.contourf(xx, yy, Y, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_train[:,0], X_train[:,1:2], color=colors[y_train].tolist(), s=10)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Frontières de décision")
plt.show()