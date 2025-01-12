# Structure From Motion and Bundle Adjustment

---

Captured high-resolution images of a JBL speaker from multiple angles for 3D reconstruction.
Applied a structure-from-motion (SfM) pipeline involving: [Feature detection and matching, Camera pose estimation, Sparse point cloud generation and Dense reconstruction]
Visualized and refined the point clouds in MeshLab, removing noise and improving model accuracy.
Converted the refined point cloud into a 3D mesh, textured and optimized in Blender.
Imported the final digital twin into Gazebo and Isaac Sim environments for simulation and analysis.

--

### Steps to Execute

1. Clone the repository as ```https://github.com/robosac333/Structure-from-Motion-and-Bundle-Adjustments.git```
2. ```cd Structure-from-Motion-and-Bundle-Adjustments```. The directory for the image directory (Line 30), along with the camera parameters (Line 16) can be updated accordingly.
3. Run ```python3 sfm.py```.
4. If executed successfully, open ```sparse.ply``` to analyse the sparse reconstruction using meshlab.

### Pipeline
1. Acquire the first image pair.
2. Detection of features using SIFT.
3. Feature matching using brute force KNN. Good feature matches are by taking the distance ratio (According to Lowe's paper) as 0.7.
4. Calculation of Essential matrix, to relate the camera locations. Outliers are rejected using RANSAC.
5. Equivalent rotation matrix (R) and translation vector (t) are taken from essential matrix using SVD.
6. Projection matrix for each camera location is calculated, and triangulation of point correspondences are calculated.
7. The correctness of triangulated points is analysed using re-projection error. The triangulated points are re - mapped onto the image plane and the deviation between the matching points is calculated. (Note that in the code, rotation matrix is converted into vector using Rodrigues equation). This will be the base point cloud, onto which newly registered images will be added.
8. A new image is taken into consideration, which shall be registered using Perspective - n - Point (PnP). For this, we need the 3D - 2D correspondence for the new image. So, the features common for image 2 and 3 are taken into consideration and only those points are taken for PnP which are visible in the newly added image (data association). After PnP, we get the world pose estimate of the image.
9. This image can now see new points, which were not there in original point cloud. So, triangulation is done for the same. Again the reprojection error is calculated.
10. Now, for each newly added image, the pipeline will repeat from step 8.

### Dataset

The dataset used is a JBL Speaker ([Link](https://drive.google.com/drive/folders/16r0MLKJSryVjavbvIiSIOnX3kfUCeG8_)). All the images have been used for obtaining the sparse point cloud.

A sample image:

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/user-attachments/assets/d7f318a0-a1d2-4d0f-8a02-4abd00f62118" alt="Image 1" width="45%" />
  <img src="https://github.com/user-attachments/assets/8a6898d6-74ca-416e-8e62-b8d435dde0ba" alt="Image 2" width="45%" />
</div>

![1734672723911](https://github.com/user-attachments/assets/45287d0f-4a5e-42aa-beaa-3aecf3b7df3f)

![1734668970073](https://github.com/user-attachments/assets/e4b1184e-af64-4c5a-b752-a27b4c2b4cec)

### Output

![Output_Point_cloud](Result/sfm.gif)


### Team Members

* [Sachin Ramesh Jadhav](https://github.com/robosac333)
* [Kautilya Reddy](https://github.com/1412kauti)
* [Kashif Ansari](https://github.com/kashifansaricodes)
* [Navdeep Singh](https://github.com/syzygy21)

IMPORTANT: Due to the lack of time, it wasn't possible to extend this project. Maybe in the near future, we would be able to optimize bundle adjustment, increase data association size, and incorporate Multiview Stereo. This is purely a project to learn and understand 3D Reconstruction of large scale data, and implement in an understandable manner, using python. Do NOT use it for research purposes. Use other incremental SfM pipelines like COLMAP.
