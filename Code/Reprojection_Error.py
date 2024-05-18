import numpy as np
import cv2

# Calculation for Reprojection error in main pipeline
def ReprojectionError(X, pts, Rt, K, homogenity):
    total_error = 0
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    r, _ = cv2.Rodrigues(R)
    if homogenity == 1:
        X = cv2.convertPointsFromHomogeneous(X.T)

    p, _ = cv2.projectPoints(X, r, t, K, distCoeffs=None)
    p = p[:, 0, :]
    p = np.float32(p)
    pts = np.float32(pts)
    if homogenity == 1:
        total_error = cv2.norm(p, pts.T, cv2.NORM_L2)
    else:
        total_error = cv2.norm(p, pts, cv2.NORM_L2)
    pts = pts.T
    tot_error = total_error / len(p)
    #print(p[0], pts[0])

    return tot_error, X, p

# Calculation of reprojection error for bundle adjustment
def OptimReprojectionError(x):
	Rt = x[0:12].reshape((3,4))
	K = x[12:21].reshape((3,3))
	rest = len(x[21:])
	rest = int(rest * 0.4)
	p = x[21:21 + rest].reshape((2, int(rest/2)))
	X = x[21 + rest:].reshape((int(len(x[21 + rest:])/3), 3))
	R = Rt[:3, :3]
	t = Rt[:3, 3]
	
	total_error = 0
	
	p = p.T
	num_pts = len(p)
	error = []
	r, _ = cv2.Rodrigues(R)
	
	p2d, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	p2d = p2d[:, 0, :]
	#print(p2d[0], p[0])
	for idx in range(num_pts):
		img_pt = p[idx]
		reprojected_pt = p2d[idx]
		er = (img_pt - reprojected_pt)**2
		error.append(er)
	
	err_arr = np.array(error).ravel()/num_pts
	
	# print(np.sum(err_arr))
	#err_arr = np.sum(err_arr)
	
	return err_arr