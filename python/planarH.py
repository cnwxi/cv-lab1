import cv2
import numpy as np


def computeH(x1, x2):
    # Q3.6
    # Compute the homography between two sets of points
    tmp = x1.shape[0]
    tmp_list = []
    for i in range(tmp):
        tmp_list.append([-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x1[i, 0] * x2[i, 0], x1[i, 0] * x2[i, 1], x1[i, 0]])
        tmp_list.append([0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x1[i, 1] * x2[i, 0], x1[i, 1] * x2[i, 1], x1[i, 1]])
    tmp_array = np.array(tmp_list)
    _, _, v = np.linalg.svd(tmp_array, full_matrices=True)
    if v[-1, -1] != 0:
        v = v[-1, :] / v[-1, -1]
    else:
        v = v[-1, :]
    H2to1 = v.reshape(3, 3)

    return H2to1


def computeH_norm(x1, x2):
    # Q3.7
    # Compute the centroid of the points
    length = x1.shape[0]
    x1_mean = np.mean(x1, axis=0)
    x2_mean = np.mean(x2, axis=0)
    # Shift the origin of the points to the centroid
    x1_shift = x1 - x1_mean
    x2_shift = x2 - x2_mean
    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_norm = np.max(np.linalg.norm(x1_shift, axis=1))
    x2_norm = np.max(np.linalg.norm(x2_shift, axis=1))
    x1_sqrt = np.sqrt(2) / x1_norm
    x2_sqrt = np.sqrt(2) / x2_norm
    x1_shift = x1_shift * x1_sqrt
    x2_shift = x2_shift * x2_sqrt
    # Similarity transform 1
    t1 = np.asarray([[x1_sqrt, 0, -x1_sqrt * x1_mean[0]],
                     [0, x1_sqrt, -x1_sqrt * x1_mean[1]],
                     [0, 0, 1]])
    # Similarity transform 2
    t2 = np.asarray([[x2_sqrt, 0, -x2_sqrt * x2_mean[0]],
                     [0, x2_sqrt, -x2_sqrt * x2_mean[1]],
                     [0, 0, 1]])
    # Compute homography
    H2to1 = computeH(x1_shift, x2_shift)
    # Denormalization
    H2to1 = np.matmul(np.linalg.inv(t1), H2to1)
    H2to1 = np.matmul(H2to1, t2)
    return H2to1


def computeH_ransac(x1, x2):
    # Q3.8
    # Compute the best fitting homography given a list of matching points
    inliers = []
    iterations = 1000
    inlier_tmp = 0
    tor = 5
    bestH2to1 = None
    for i in range(iterations):
        idx = np.arange(0, len(x1))
        np.random.shuffle(idx)
        ptsel = idx[0:4]
        H2to1 = computeH_norm(x1[ptsel], x2[ptsel])
        x2_ = np.hstack((x2, np.ones(len(x2)).reshape(-1, 1)))
        x1_ = np.matmul(H2to1, x2_.T)[0:2, :]
        dist = np.linalg.norm(x1.T - x1_, axis=0)
        inlier = (dist < tor).astype(int)
        inlier_cont = np.sum(inlier)
        if inlier_cont > inlier_tmp:
            inliers = inlier
            bestH2to1 = H2to1
            inlier_tmp = inlier_cont
    if len(inliers) > 1:
        index = [i for i, j in enumerate(inliers) if j == 1]
        if len(index) >= 4:
            bestH2to1 = computeH_norm(x1[index], x2[index])
    else:
        bestH2to1 = np.eye(3)
    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    mask_temp = np.ones((template.shape[0], template.shape[1]), dtype=np.uint8)
    mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # Warp mask by appropriate homography
    composite_img = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))
    # Warp template by appropriate homography
    mask_temp_warp = cv2.warpPerspective(mask_temp, H2to1, (mask_img.shape[1], mask_img.shape[0]))
    # Use mask to combine the warped template and the image
    warp_not = cv2.bitwise_not(mask_temp_warp) // 255
    warp_not = np.stack([warp_not, warp_not, warp_not])
    warp_not = np.transpose(warp_not, (1, 2, 0))
    background = img * warp_not
    composite_img = background + composite_img
    return composite_img
