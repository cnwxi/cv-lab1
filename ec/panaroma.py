import cv2
import matplotlib.pyplot as plt

# Import necessary functions
from python.matchPics import matchPics
from python.planarH import computeH_ransac

# Write script for Q4.2x
img1 = cv2.imread('../data/pano_left.jpg')
img2 = cv2.imread('../data/pano_right.jpg')
# img1 = cv2.imread('../data/left.png')
# img2 = cv2.imread('../data/right.png')
matches, locs1, locs2 = matchPics(img1, img2)
locs1 = locs1[matches[:, 0]]

matches, locs1, locs2 = matchPics(img1, img2)
locs1 = locs1[matches[:, 0]]
locs1 = locs1[:, [1, 0]]
locs2 = locs2[matches[:, 1]]
locs2 = locs2[:, [1, 0]]

bestH2to1, inliers = computeH_ransac(locs1, locs2)

r, c = int(2 * img1.shape[1]), int(1.5 * img1.shape[0])

output = cv2.warpPerspective(img2, bestH2to1, (r, c))
output[0:img1.shape[0], 0:img1.shape[1]] = img1

cv2.imwrite('../data/pano_out.jpg', output)
for i, element in enumerate([img1, img2, output]):
    plt.subplot(3, 1, 1 + i)
    plt.imshow(cv2.cvtColor(element, cv2.COLOR_BGR2RGB))

plt.show()
