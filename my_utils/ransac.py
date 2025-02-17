import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression



def ransac(points):
    '''
    args:
        points: list of points in the form of [x,y,z]
    
    returns:
        inlier_mask: boolean mask for inliers
    
    '''

    points= np.array(points)

    # Separate into X (features) and y (target)
    X = points[:, :2]  # Take the x and y coordinates as input features
    y = points[:, 2]   # Use z as the output

    # Create a RANSAC model
    ransac = RANSACRegressor(base_estimator=LinearRegression(), residual_threshold=0.01)
    model = make_pipeline(PolynomialFeatures(degree=1), ransac)

    # Fit the model to data
    model.fit(X, y)

    # Get the inliers and outliers
    inlier_mask = ransac.inlier_mask_  # Boolean mask for inliers
    outlier_mask = np.logical_not(inlier_mask)
    return inlier_mask

