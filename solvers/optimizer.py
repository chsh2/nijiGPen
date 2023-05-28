import numpy as np

def pca(data):
    """
    Calculate the matrix that projects 3D shapes to 2D in the optimal way
    """
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    idx = eigenvalues.argsort()[::-1]  
    values =  eigenvalues[idx]
    mat = eigenvectors[:,idx]
    return mat, values