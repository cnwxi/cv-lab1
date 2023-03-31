import cv2
import matplotlib.pyplot as plt
import numpy as np

from python.planarH import compositeH

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
hp_cover = cv2.resize(hp_cover, ((cv_cover.shape[1]), cv_cover.shape[0]))

orb = cv2.ORB_create()
bf = cv2.BFMatcher()
indexParams = dict(algorithm=0, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

kp1, desc1 = orb.detectAndCompute(cv_cover, None)
kp2, desc2 = orb.detectAndCompute(cv_desk, None)

desc1 = desc1.astype(np.float32)
desc2 = desc2.astype(np.float32)

matches = bf.match(desc1, desc2)
max_dis = matches[0].distance
min_dis = matches[0].distance
good = []
for i in matches:
    min_dis = min(min_dis, i.distance)
    max_dis = max(max_dis, i.distance)
print('min', min_dis)
print('max', max_dis)
for i in matches:
    if i.distance <= max(1.36 * min_dis, 0.5 * max_dis):
        good.append(i)

matches = good

img = cv2.drawMatches(cv_cover, kp1, cv_desk, kp2, matches, None, flags=2)

locs1 = np.array([[kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1]] for i in matches])
locs2 = np.array([[kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1]] for i in matches])

M, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC)

comp_img = compositeH(M, hp_cover, cv_desk)
plt.suptitle('HappyPotter_orb.py')
plt.subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 1, 2)
plt.imshow(cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB))
plt.show()
