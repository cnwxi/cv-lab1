import cv2
import matplotlib.pyplot as plt
import numpy as np

from python.planarH import compositeH

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
hp_cover = cv2.resize(hp_cover, ((cv_cover.shape[1]), cv_cover.shape[0]))

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
indexParams = dict(algorithm=0, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

kp1, desc1 = sift.detectAndCompute(cv_cover, None)
kp2, desc2 = sift.detectAndCompute(cv_desk, None)

desc1 = desc1.astype(np.float32)
desc2 = desc2.astype(np.float32)

matches = bf.knnMatch(desc1, desc2, k=2)

good_match = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_match.append([m])

matches = [i[0] for i in good_match]
print(len(matches))
img = cv2.drawMatchesKnn(cv_cover, kp1, cv_desk, kp2, good_match, None, flags=2)

locs1 = np.array([[kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1]] for i in matches])
locs2 = np.array([[kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1]] for i in matches])

M, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC)

comp_img = compositeH(M, hp_cover, cv_desk)

plt.subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 1, 2)
plt.imshow(cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB))
plt.show()
