import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

imagearray = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
plt.imshow(imagearray[200,:,:])
plt.show()