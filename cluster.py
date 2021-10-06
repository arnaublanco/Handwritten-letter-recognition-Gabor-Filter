import pdb
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cluster(image):

	clusters = np.zeros((image.shape[0],image.shape[1]))
	c = 1
	thr = 1
	arr = np.arange(-thr,thr,1)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i][j] == 255:
				for m in arr:
					for n in arr:
						if image[i+m][j+n] == 255:


	pdb.set_trace()
	clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=1).fit(tmp)
	pdb.set_trace()
	#labels = clustering.labels_.reshape((image.shape[0],image.shape[1]))

	return clusters