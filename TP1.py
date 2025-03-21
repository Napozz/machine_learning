import numpy as np

# Question 1 :
data = np.loadtxt("dataset.dat")
X = data[:, :2]
y = data[:, 2] #Le fait de ne pas mettre 2: permet de faire de y une 'liste' plutot qu'un tableau de une colonne
y = y.astype(int)

# Question 2 :
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state=1)

# Question 3 :
from matplotlib import pyplot
colors = np.array([x for x in "rgbcmyk"])
pyplot.scatter(X[:,0],X[:,1], color=colors[y].tolist(), s=10)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.show()

# Question 5 :
from sklearn.neighbors import KNeighborsClassifier
one_NN = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
one_NN.fit(X_train, y_train)

#score sur la base d'apprentissage
acc_train_set = one_NN.score(X_train, y_train)
#score sur la base de test
acc_test_set =  one_NN.score(X_test, y_test)

print('Accuracy on training set:', acc_train_set)
print('Accuracy on test set:', round(acc_test_set, 2))
print(one_NN.get_params())

# Question 6
y_pred_test = one_NN.predict(X_test)

#matrice de confusion
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)

print(confusion_matrix)
print(confusion_matrix.sum())

# Question 7
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=one_NN.classes_)
disp.plot()
pyplot.show()

# Question 8
from matplotlib import pyplot
colors = np.array([x for x in "rgbcmyk"])

# Créer une grille
x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
np.arange(y_min, y_max, y_h))
Y = one_NN.predict(np.c_[xx.ravel(), yy.ravel()])
Y = Y.reshape(xx.shape)
pyplot.grid()

#afficher les frontières/données d'apprentissage
pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_train[:,0], X_train[:,1], color=colors[y_train].tolist(), s=10)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.title("Frontières de décision")
pyplot.show()

# Question 9
from matplotlib import pyplot

colors = np.array([x for x in "rgbcmyk"])

# Créer une grille
x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
np.arange(y_min, y_max, y_h))
Y = one_NN.predict(np.c_[xx.ravel(), yy.ravel()])
Y = Y.reshape(xx.shape)
pyplot.grid()

#afficher les frontières/données d'apprentissage
pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_test[:,0], X_test[:,1], color=colors[y_test].tolist(), s=10)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.title("Frontières de décision")
pyplot.show()

# Question 10
#impact de la taille de la base d'apprentissage
acc_test = []
abscisses = []
for size in range(1,100):
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, train_size=size/100, random_state=1)
    one_NN.fit(X_train1, y_train1)
    #score sur la base de test
    acc_test.append(one_NN.score(X_test, y_test))
    abscisses.append(size)

print(len(acc_test))


# Question 11
pyplot.plot(abscisses, acc_test, linewidth=2.0)
pyplot.xlabel("Nombres d'exemples")
pyplot.ylabel("Taux de reconnaissance")
pyplot.grid()
pyplot.show()


# Question 12


# Question 13
#impact de la taille de la base de test
"""acc_test = []
abscisses = []
for size in range(1,90):
    X_train2, X_test2, y_test2, y_train2 = train_test_split(X_test, y_test, train_size=size/90,random_state=1)
    one_NN.fit(X_train, y_train)
    #score sur la base de test
    acc_test.append(one_NN.score(X_test2, y_test2))
    abscisses.append(size)

# Question 14
pyplot.plot(abscisses, acc_test, linewidth=2.0)
pyplot.xlabel("Nombres d'exemples")
pyplot.ylabel("Taux de reconnaissance")
pyplot.grid()
pyplot.show()"""

# Question 15

# Question 16

acc_train = []
acc_test = []
erreur_acc_apprentissage = []
erreur_acc_test = []
kmax = len(X_train)-1 # kmax = on prend tous les voisins possibles
liste_k = []
for k in range (1, kmax):
    k_NN = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    k_NN.fit(X_train, y_train)
    taux_erreur_apprentissage = 1 - k_NN.score(X_train, y_train) 
    acc_train.append(1 - taux_erreur_apprentissage)
    erreur_acc_apprentissage.append(taux_erreur_apprentissage)
    taux_erreur_test = 1 - k_NN.score(X_test, y_test)
    acc_test.append(1 - taux_erreur_test)
    erreur_acc_test.append(taux_erreur_test)
    liste_k.append(k)

print(len(acc_test))

# Question 17
# kmax = on prend tous les voisins possibles
# i.e. toutes les valeurs sauf le point qu'on cherche à classer

# question 18
pyplot.plot(liste_k, erreur_acc_test, linewidth=2.0)
pyplot.xlabel("k")
pyplot.ylabel("Taux d'erreur'")
pyplot.grid()
pyplot.show()

# Question 19
# k* environ 24 d'après le graph

# Question 20
one_NN = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
one_NN.fit(X_train, y_train)
# Créer une grille
x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
np.arange(y_min, y_max, y_h))
Y = one_NN.predict(np.c_[xx.ravel(), yy.ravel()])
Y = Y.reshape(xx.shape)
pyplot.grid()

pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_train[:,0], X_train[:,1], color=colors[y_train].tolist(), s=10)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.title("Frontières de décision pour k=1")
pyplot.show()

# Question 21
# On a vu que k* = 21 ou 23
kstar_NN = KNeighborsClassifier(n_neighbors=21, algorithm='brute')
kstar_NN.fit(X_train, y_train)
# Créer une grille
x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
np.arange(y_min, y_max, y_h))
Y = kstar_NN.predict(np.c_[xx.ravel(), yy.ravel()])
Y = Y.reshape(xx.shape)
pyplot.grid()

pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_train[:,0], X_train[:,1], color=colors[y_train].tolist(), s=10)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.title("Frontières de décision pour k=k*")
pyplot.show()

# Question 22
kmax_NN = KNeighborsClassifier(n_neighbors=199, algorithm='brute')
kmax_NN.fit(X_train, y_train)
# Créer une grille
x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
np.arange(y_min, y_max, y_h))
Y = kmax_NN.predict(np.c_[xx.ravel(), yy.ravel()])
Y = Y.reshape(xx.shape)
pyplot.grid()

pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_train[:,0], X_train[:,1], color=colors[y_train].tolist(), s=10)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.title("Frontières de décision pour k=kmax")
pyplot.show()

# Question 25
pyplot.plot(liste_k, erreur_acc_apprentissage, linewidth=2.0)
pyplot.xlabel("k")
pyplot.ylabel("Taux d'erreur en apprentissage")
pyplot.grid()
pyplot.show()