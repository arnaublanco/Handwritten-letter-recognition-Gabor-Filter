import joblib
import numpy as np
from skimage.filters import gabor
from sklearn.decomposition import PCA

def computeFeatures(img,pca):
	curr = []
	for freq in np.arange(1,11,2):
	    for orientation in range(0,360,45):
	        filt_real, filt_imag = gabor(img,frequency=freq,theta=orientation) # Compute Gabor Filter on image i for a given scale and orientation
	        phase = np.angle(filt_real + 1.0j*filt_imag) # Compute phase with real and complex part
	        curr = np.append(curr,phase.reshape([1,phase.size])) # Feature reduction
	        
	features = pca.transform(curr.reshape([1,curr.size]))
	return features