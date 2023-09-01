# Handwritten letter recognition

In this project, we utilize the MNIST dataset as a foundation for the classification of letters using a Support Vector Machine (SVM). The workflow can be summarized in the following key steps:

1. **Data Import:** The project commences with the importation of the <a href="http://yann.lecun.com/exdb/mnist/">MNIST dataset</a>, which is a widely recognized collection of handwritten digits. This dataset serves as the basis for our letter classification task.

2. **Gabor Filter Computation:** To enhance the discriminative power of the dataset, Gabor filters are computed for each image. Gabor filters are valuable in capturing texture and edge information in images, and their application here aids in accentuating critical features within handwritten letters. <a href="https://en.wikipedia.org/wiki/Gabor_filter">See here</a>.

3. **Dimensionality Reduction with PCA:** Given the potentially high-dimensional nature of the Gabor-filtered data, Principal Component Analysis (PCA) is employed as a dimensionality reduction technique. PCA helps in condensing the data while retaining its essential characteristics. By reducing dimensionality, we mitigate computational complexity for the classifier.

4. **SVM Classification:** Finally, the reduced-dimension data is fed into the Support Vector Machine (SVM) classifier. SVM is a robust machine learning algorithm known for its effectiveness in binary and multiclass classification tasks. In our case, it excels at distinguishing between different letters based on the extracted features.

By combining these steps, we create a comprehensive pipeline for letter classification, starting with data preprocessing, enhancing feature extraction with Gabor filters, optimizing efficiency with PCA, and concluding with accurate classification through SVM. This approach allows us to tackle the MNIST dataset's letter classification challenge effectively.

The training can be found in `trainModel.py`.
