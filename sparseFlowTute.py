
import os
from glob import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt

from motion_detection_utils import *

#####

# fpath = r"testImgs"
# fpath = r"out_frames"
fpath = r"out_frames_2"

# image_paths = sorted(glob(f"{fpath}/*.JPG"))
image_paths = sorted(glob(f"{fpath}/*.jpg"))
print(len(image_paths))

idx = 1
frame1 = cv2.cvtColor(cv2.imread(image_paths[idx]), cv2.COLOR_BGR2RGB)
frame2 = cv2.cvtColor(cv2.imread(image_paths[idx + 1]), cv2.COLOR_BGR2RGB)

_, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].imshow(frame1)
ax[0].set_title("frame 1")
ax[1].imshow(frame2)
ax[1].set_title("frame 2")

def motion_comp(prev_frame, curr_frame, num_points=500, points_to_use=500, transform_type='affine'):
    """ Obtains new warped frame1 to account for camera (ego) motion
        Inputs:
            prev_frame - first image frame
            curr_frame - second sequential image frame
            num_points - number of feature points to obtain from the images
            points_to_use - number of point to use for motion translation estimation 
            transform_type - type of transform to use: either 'affine' or 'homography'
        Outputs:
            A - estimated motion translation matrix or homography matrix
            prev_points - feature points obtained on previous image
            curr_points - feature points obtaine on current image
        """
    transform_type = transform_type.lower()
    assert(transform_type in ['affine', 'homography'])

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

    # get features for first frame
    corners = cv2.goodFeaturesToTrack(prev_gray, num_points, qualityLevel=0.01, minDistance=10)

    # get matching features in next frame with Sparse Optical Flow Estimation
    matched_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)

    # reformat previous and current corner points
    prev_points = corners[status==1]
    curr_points = matched_corners[status==1]

    # sub sample number of points so we don't overfit
    if points_to_use > prev_points.shape[0]:
        points_to_use = prev_points.shape[0]

    index = np.random.choice(prev_points.shape[0], size=points_to_use, replace=False)
    prev_points_used = prev_points[index]
    curr_points_used = curr_points[index]

    # find transformation matrix from frame 1 to frame 2
    if transform_type == 'affine':
        A, _ = cv2.estimateAffine2D(prev_points_used, curr_points_used, method=cv2.RANSAC)
    elif transform_type == 'homography':
        A, _ = cv2.findHomography(prev_points_used, curr_points_used)

    return A, prev_points, curr_points

A, prev_points, curr_points = motion_comp(frame1, frame2, num_points=10000, points_to_use=10000, transform_type='affine')
# warp frame 1 to account for camera motion
transformed1 = cv2.warpAffine(frame1, A, dsize=(frame1.shape[:2][::-1])) # affine transform
# transformed1 = cv2.warpPerspective(frame1, A, dsize=(frame1.shape[:2][::-1])) # homography transform

_, ax2 = plt.subplots(1, 2, figsize=(15, 10))
ax2[0].imshow(transformed1)
ax2[0].set_title("warped frame 1")
ax2[1].imshow(frame2)
ax2[1].set_title("frame 2")

og_delta = cv2.subtract(frame2, frame1)
comped_delta = cv2.subtract(frame2, transformed1)

_, ax3 = plt.subplots(1, 2, figsize=(15, 10))
ax3[0].imshow(og_delta)
ax3[0].set_title("Frame Difference of Original Frames")
ax3[1].imshow(comped_delta)
ax3[1].set_title("Frame Difference of Motion Comped Frames")

plt.figure()
plt.imshow(cv2.cvtColor(comped_delta, cv2.COLOR_RGB2GRAY) > 50)

plt.figure()
img = plot_points(frame2.copy(), curr_points)

# affine
A = np.vstack((A, np.zeros((3,)))) # get 3x3 matrix to xform points
compensated_points = np.hstack((prev_points, np.ones((len(prev_points), 1)))) @ A.T 

# homography
# warped_points = np.hstack((prev_points, np.ones((len(prev_points), 1)))) @ A.T

compensated_points = compensated_points[:, :2]

print(f" Prev Key Points: {np.round(prev_points[10], 2)} \n",
      f"Compensated Key Points: {np.round(compensated_points[10], 2)} \n",
      f"Current Key Points: {np.round(curr_points[10], 2)}")

flow = curr_points - prev_points
compensated_flow = curr_points - compensated_points

num_bins = 200
fig, ax4 = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
fig.suptitle("Flow Vector Histograms")
ax4[0][0].hist(flow[:, 0], bins=num_bins) # curr_points[:, 0] - prev_points[:, 0], bins=num_bins);
ax4[0][0].set_title("Raw Flow Vector Histogram (Horiztonal)")
ax4[0][1].hist(compensated_flow[:, 0], bins=num_bins)
ax4[0][1].set_title("Compensated Flow Vector Histogram (Horiztonal)")

ax4[1][0].hist(flow[:, 1], bins=num_bins)
ax4[1][0].set_title("Raw Flow Vector Histogram (Vertical)")
ax4[1][1].hist(compensated_flow[:, 1], bins=num_bins)
ax4[1][1].set_title("Compensated Flow Vector Histogram (Vertical)")

import scipy
from scipy.stats import kurtosis

c = 2 # tunable scale factor

# Obtain distance metric for displacement
# x = np.sum(np.abs(compensated_flow), axis=1) # l1 norm
x = np.linalg.norm(compensated_flow, ord=2, axis=1) # l2 norm

# get various statistics
mu = np.mean(x)
sig = np.std(x, ddof=1) # unbiased estimator?
med = np.median(x)
mad = np.median(np.abs(x - med)) # Median Absolute Deviation
iqr = scipy.stats.iqr(x)
q1 = np.percentile(x, 25)
q3 = np.percentile(x, 75)

# get Kurtosis
# We expect a very Leptokurtic distribution with extrememly long tails noting the outliers
k = kurtosis(x, bias=False, fisher=True)

# if distriubtion doesn't have too long of tails reduce outlier threshold parameter 
if k < 1:
    c /= 2
    

# upper and lower bounds for outlier detection
upper_bound = mu + c*sig
lower_bound = mu - c*sig

# display statistics
print(f"Sample Mean: {mu :.4f}, Sample Std Dev: {sig :.4f} IQR: {iqr :.4f}\n"
      f"Sample Median: {med :.4f} MAD: {mad :.4f} \n"
      f"Upper Outlier Bound: {upper_bound :.4f} Lower Outlier Bound: {lower_bound :.4f} \n"
      f"Kurtosis: {k :.4f}")

l1 = np.sum(np.abs(compensated_flow), axis=1) # l1 norm
l2 = np.linalg.norm(compensated_flow, ord=2, axis=1) # l2 norm

_, ax5 = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
ax5[0].hist(l1, bins=100)
ax5[0].set_title("Histrogram of L1 Distance of Compensated Flow Vectors")
ax5[1].hist(l2, bins=100)
ax5[1].set_title("Histrogram of L2 Distance of Compensated Flow Vectors")


lap_dist = lambda x, b, mu : (1/(2*b)) * np.exp(-np.abs(x - mu)/b)
# log_normal = lambda x, mu, sig : (1/(x*sig*np.sqrt(2*np.pi)) * np.exp(-np.log(x - mu)**2 / (2*sig**2)) )
# b = sig/np.sqrt(2)
b = np.sqrt(sig/2) # gives a better fit
vals = np.arange(x.min(), x.max(), 0.1)

plt.figure()
plt.hist(x, bins=100, density=True)
# plt.plot(vals, lap_dist(vals, b, mu), label="Laplace")
plt.title("Histrogram of L2 Norm of Compensated Flow Vectors")
plt.ylabel("Counts")
plt.xlabel("Bins (L2 distance)")
plt.legend()

motion_idx = (x >= upper_bound)
print(motion_idx.sum())

motion_points = curr_points[motion_idx]
motion_flow = compensated_flow[motion_idx]
img = plot_points(frame2.copy(), motion_points, radius=10)

plt.figure()
plt.imshow(img)
plt.title("Motion Points on Frame 2")

motion = compensated_points[motion_idx] - curr_points[motion_idx] # curr_points[idx] - compensated_points[idx]
magnitude = np.linalg.norm(motion, ord=2, axis=1)
angle = np.arctan2(motion[:, 0], motion[:, 1]) # horizontal/vertial

X = np.hstack((motion_points, np.c_[magnitude], np.c_[angle]))
print(X.shape)

from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=30.0, min_samples=2)
print(clustering.fit(X))
print(np.unique(clustering.labels_))

img = frame2.copy()
for i, lbl in enumerate(clustering.labels_):
    if lbl >= 0:
        color = get_color((i+1)*1)
        img = plot_points(img, motion_points[clustering.labels_ == lbl], radius=30, color=color)

plt.figure()
plt.imshow(img)
plt.title("Unfiltered Motion Clusters on Frame 2")

lbl = 0
img = plot_points(frame2.copy(), motion_points[clustering.labels_ == lbl], radius=20)

plt.figure()
plt.imshow(img)

h, w, _ = frame1.shape
w,h

# set angle uniformity threshold
angle_thresh = 0.15 # angular threshold in radians
edge_thresh = 50 # threshol of which to remove clusters that appear on the edges
cluster_labels = [] # clusters to keep
clusters = []
for lbl in np.unique(clustering.labels_):

    angle_std = angle[clustering.labels_ == lbl].std(ddof=1)
    if angle_std <= angle_thresh:
        cluster = motion_points[clustering.labels_ == lbl]

        # remove clusters that are too close to the edges
        centroid = cluster.mean(axis=0)
        if not (np.any(centroid < edge_thresh) or np.any(centroid > np.array([w - edge_thresh, h - edge_thresh]))):
            cluster_labels.append(lbl)
            clusters.append(cluster)

    # TEMP
    # print(np.degrees(np.arctan2(delta[:,0], delta[:,1])).std())
    # print(np.arctan2(delta[:,0], delta[:,1]).std())
    # print()

print(cluster_labels)

img = frame2.copy()
for i, cluster in enumerate(clusters):
    color = get_color((i+1)*5)
    img = plot_points(img, cluster, radius=20, color=color)

plt.figure()
plt.imshow(img)
plt.title("Detected Motion Clusters on Frame 2")


plt.show()