import matplotlib as plt
import numpy as np
import make_classification as mc
import LinearSVC as LSVC

#SVC parameters
n = 10000
l_rate = 0.01
C = 1


#Classification parameters
u = 1
n = 100
d = 2
X_train,y_train, X_test, y_test, a = mc.make_classification(d,n,u)

SVC_instance = LSVC.LinearSVC(eta=l_rate, n_iter=n, random_state=1)
SVC_instance.fit(X_train,y_train,C)