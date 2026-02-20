import matplotlib.pyplot as plt
import numpy as np
import make_classification as mc
import LinearSVC as LSVC
import MarginBand as mb

#SVC parameters
n = 100
l_rate = 0.0001
C = 200


#Classification parameters
u = 1
n_samples = 1000
d = 2
X_train,y_train, X_test, y_test, a = mc.make_classification(d,n_samples,u)

SVC_instance = LSVC.LinearSVC(eta=l_rate, n_iter=n, random_state=1)
SVC_instance.fit(X_train,y_train,C)


y_test_predict = SVC_instance.predict(X_test)

# prediction class errors
plt.figure()
plt.plot(SVC_instance.class_errors_)
plt.xlabel("Data")
plt.ylabel("Class Errors")
plt.title("SVC - Class Errors")
plt.savefig("images/SVC_class_errors.png")
plt.close()

# prediction margin violations
plt.figure()
plt.plot(SVC_instance.margin_violations_)
plt.xlabel("Data")
plt.ylabel("Margin Violations")
plt.title("SVC - Margin Violations")
plt.savefig("images/SVC_margin_violations.png")
plt.close()

# prediction losses
plt.figure()
plt.plot(SVC_instance.losses_)
plt.xlabel("Data")
plt.ylabel("Loss")
plt.title("SVC - Loss")
plt.savefig("images/SVC_loss.png")
plt.close()

#plot of actual plane vs prediction
#Separation Plane
x_plot = np.linspace(-u,u,100)
y_plot = -a[0]/a[1] * x_plot
plt.plot(x_plot,y_plot)

#Data and associated prediction
colors = np.where(y_test_predict > 0, 'r','b')
plt.scatter(
    X_test[:,0], X_test[:,1],
    c = colors)
plt.ylim(-u,u)
plt.title("prediction")
plt.savefig("images/SVC_prediction.png")
plt.close()

#Margin Band
mb.plot_svm_margin(X_test,y_test,SVC_instance)
plt.ylim(-u,u)
plt.savefig("images/SVC_band.png")
plt.close()