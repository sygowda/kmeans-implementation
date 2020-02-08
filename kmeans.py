import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.

    _, m = x.shape
    centers = np.zeros(n_cluster, dtype=int)
    r = np.floor(generator.rand() * 11).astype(int)
    centers[0] = r

    for i in range(1, n_cluster):
        old_center = x[centers[i - 1], :]
        dist = Euclidean_distance_prob(old_center, x)
        idx = random_plate(generator.rand(), dist)
        centers[i] = idx
    centers = centers.tolist()

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers


def Euclidean_distance_prob(center, x):
    d = np.sum((x - center) ** 2, axis=1) ** 0.5
    p = d / np.sum(d)
    return np.cumsum(p)


def random_plate(random_number, probs):
    n = len(probs)
    for i in range(n):
        if random_number <= probs[i]:
            return i
        else:
            continue


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        """
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        """
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #   raise Exception('Implement fit function in KMeans class')

        y = np.zeros(N)
        centroids = x[self.centers]
        previous = centroids
        old_J = self.obj_value(x, y, centroids)
        for i in range(self.max_iter):
            y = decide(centroids, x)
            centroids = self.update_centers(x, y, centroids)
            J = self.obj_value(x, y, centroids)
            if self.convergent(old_J, J):
                break
            else:
                old_J = J
                continue
            self.max_iter = i + 1

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

    def obj_value(self, x, y, centroids):
        J = 0
        for i in range(self.n_cluster):
            idx = np.argwhere(y == i)
            idx = idx.reshape(len(idx))
            if len(idx) > 0:
                data = x[idx]
                J += np.sum(Euclidean_distance(centroids[i], data) ** 2)
        return J

    def update_centers(self, x, y, centroids):
        N, D = x.shape
        for i in range(self.n_cluster):
            idx = np.argwhere(y == i)
            idx = idx.reshape(len(idx))
            if len(idx) > 0:
                data = x[idx]
                centroids[i] = data.mean(axis=0)
        return centroids

    def convergent(self, oJ, J):
        diff = np.abs(oJ - J)
        if self.e >= diff:
            return True
        else:
            return False


def Euclidean_distance(center, x):
    dist = np.sum((x - center) ** 2, axis=1) ** 0.5
    return dist


def decide(center, x):
    N, D = x.shape
    n = len(center)
    dist = []
    for i in center:
        d = Euclidean_distance(i, x)
        dist.append(d)
    dist = np.concatenate(dist).reshape(n, N)
    classes = np.argmin(dist, axis=0)
    return classes


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        """
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        """

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        kmean = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, class_assign, iter_num = kmean.fit(x, centroid_func)
        centroid_labels = np.zeros(self.n_cluster, dtype=int)
        for i in range(self.n_cluster):
            idx = np.argwhere(class_assign == i)
            idx = idx.reshape(len(idx))
            label_i = y[idx].tolist()
            if len(label_i) > 0:
                unique_elements, counts_elements = np.unique(label_i, return_counts=True)
                centroid_labels[i] = unique_elements[np.argmax(counts_elements)]
            else:
                centroid_labels[i] = 0

        # raise Exception(
        #   'Implement fit function in KMeansClassifier class')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        """
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        """

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        cluster_assign = decide(self.centroids, x)
        labels = []
        for i in range(len(cluster_assign)):
            labels.append(self.centroid_labels[cluster_assign[i]])

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):
    """
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    """

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    h, w, l = image.shape
    x = image.reshape(h * w, 3)
    n_cluster, _ = code_vectors.shape
    assign = decide(code_vectors, x)
    new_im = np.zeros(shape=(h * w, 3))

    for i in range(h * w):
        new_im[i, :] = code_vectors[assign[i]]
    new_im = new_im.reshape(h, w, 3)

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im
