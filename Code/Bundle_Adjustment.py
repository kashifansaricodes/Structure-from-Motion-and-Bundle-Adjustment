import numpy as np
import cv2
from Reprojection_Error import OptimReprojectionError
from scipy.optimize import least_squares

def BundleAdjustment(points_3d, temp2, Rtnew, K, r_error):

	# Set the Optimization variables to be optimized
	opt_variables = np.hstack((Rtnew.ravel(), K.ravel()))
	opt_variables = np.hstack((opt_variables, temp2.ravel()))
	opt_variables = np.hstack((opt_variables, points_3d.ravel()))

	error = np.sum(OptimReprojectionError(opt_variables))
	corrected_values = least_squares(fun = OptimReprojectionError, x0 = opt_variables, gtol = r_error)
	corrected_values = corrected_values.x
	Rt = corrected_values[0:12].reshape((3,4))
	K = corrected_values[12:21].reshape((3,3))
	rest = len(corrected_values[21:])
	rest = int(rest * 0.4)
	p = corrected_values[21:21 + rest].reshape((2, int(rest/2)))
	X = corrected_values[21 + rest:].reshape((int(len(corrected_values[21 + rest:])/3), 3))
	p = p.T
	
	return X, p, Rt