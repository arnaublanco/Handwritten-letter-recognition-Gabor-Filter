# Import required libraries
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from PIL import Image
from skimage.filters import gabor
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import idx2numpy
import pdb
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier
np.seterr(divide='ignore', invalid='ignore')

# Load NMIST dataset.
trainX = idx2numpy.convert_from_file("data/train-images-idx3-ubyte")
labelsX = idx2numpy.convert_from_file("data/train-labels-idx1-ubyte")

features = np.empty((1, 2))
labels = np.empty((1,))
mxm = 40
for i in range(mxm):
	print('Round',i+1)
	if labelsX[i] == 0:
		lbl = 0
	elif labelsX[i] == 1:
		lbl = 1
	else:
		lbl = 2

	image = trainX[i]
	for freq in np.arange(0.2,1.2,0.2):
		for orientation in range(0,360,45):
			filt_real, filt_imag = gabor(image,frequency=freq,theta=orientation) # Compute Gabor Filter on image i for a given scale and orientation
			phase = np.angle(filt_real + 1.0j*filt_imag) # Compute phase with real and complex part
			pca = PCA(n_components=2) # Initialize a 2-component PCA
			pca.fit(phase) # Fit phase in PCA
			curr = pca.transform(phase) # Feature reduction
			features = np.append(features,curr,axis=0) # Append features
			labels = np.append(labels,[lbl]*len(curr)) # Append labels


kf = KFold(n_splits=mxm)
for train, test in kf.split(features):
	X_train, X_test, y_train, y_test = features[train], features[test], labels[train], labels[test]
	clf = OneVsOneClassifier(svm.SVC(kernel='rbf'))
	clf.fit(X_train,y_train)
	print('Accuracy:',clf.score(X_test,y_test))