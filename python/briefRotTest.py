import cv2
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

from helper import plotMatches
from matchPics import matchPics

# Q3.5
# Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
# cv_desk = cv2.imread('../data/cv_desk.png')

histogram_list = []
for i in tqdm(range(36)):
    # Rotate Image
    rot_cover = scipy.ndimage.rotate(cv_cover, angle=i * 10)
    # cv2.imshow('ori', cv_cover)
    # cv2.imshow('rot', rot_desk)
    # cv2.waitKey(0)
    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cv_cover, rot_cover)
    if i % 12 == 0:
        plotMatches(cv_cover, rot_cover, matches, locs1, locs2)
    # Update histogram
    histogram_list.append(len(matches))

# Display histogram
x = list(range(0, 360, 10))
plt.bar(x, histogram_list, width=4)
plt.show()
