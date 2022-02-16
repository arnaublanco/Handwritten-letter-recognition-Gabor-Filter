# Import required libraries
import numpy as np
from sklearn import svm
from skimage.filters import gabor
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import idx2numpy
import pdb
from sklearn.model_selection import RepeatedKFold
from sklearn.multiclass import OneVsOneClassifier
import joblib
import time
from sklearn.decomposition import PCA
np.seterr(divide='ignore', invalid='ignore')

# Load NMIST dataset.
trainX = idx2numpy.convert_from_file("data/train-images-idx3-ubyte")
labelsX = idx2numpy.convert_from_file("data/train-labels-idx1-ubyte")

# Compute features
mxm = 20000
features = np.empty((mxm, trainX.shape[1]*trainX.shape[2]*40))
labels = np.array([])
for i in range(mxm):
	print('Computing Gabor Filter for image',i+1)

	lbl = labelsX[i]
	image = trainX[i]
	curr = []
	for freq in np.arange(1,11,2):
		for orientation in range(0,360,45):
			filt_real, filt_imag = gabor(image,frequency=freq,theta=orientation) # Compute Gabor Filter on image i for a given scale and orientation
			phase = np.angle(filt_real + 1.0j*filt_imag) # Compute phase with real and complex part
			curr = np.append(curr,phase.reshape([1,phase.size])) # Feature reduction
	labels = np.append(labels,lbl) # Append labels

	features[i,:] = curr # Append features

# Dimensionality reduction
print('Fitting PCA...')
pca = PCA(n_components=50)
pca.fit(features)
features = pca.transform(features)
print('PCA fitted. Computing SVM...')

# Cross-validation: training and accuracy
kf = RepeatedKFold(n_splits=3, n_repeats=3)
clf = OneVsOneClassifier(svm.SVC(kernel='rbf',C=100))
counter = 1
acc = []
for train, test in kf.split(features):
	start = time.time()
	print('> Computing fold',counter)
	X_train, X_test, y_train, y_test = features[train,:], features[test,:], labels[train], labels[test]
	clf.fit(X_train,y_train)
	currAcc = clf.score(X_test,y_test)
	end = time.time()
	print('Accuracy:',round(currAcc,3),'| Elapsed time (s):',round(end-start,3))
	acc.append(currAcc)
	counter = counter + 1

# Save model in .cls file
filename_SVM = 'model.cls'
filename_PCA = 'pca.cls'
joblib.dump(clf, filename_SVM)
joblib.dump(pca, filename_PCA)
print('Overall accuracy:',round(np.mean(acc),3))