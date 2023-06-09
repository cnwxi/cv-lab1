import cv2

from python.helper import briefMatch
from python.helper import computeBrief
from python.helper import corner_detection


def matchPics(I1, I2):
    # I1, I2 : Images to match
    image1 = I1
    image2 = I2

    # Convert Images to GrayScale
    if len(image1.shape) >= 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) >= 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Detect Features in Both Images
    locs1 = corner_detection(image1, sigma=0.15)
    locs2 = corner_detection(image2, sigma=0.15)
    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(image1, locs1)
    desc2, locs2 = computeBrief(image2, locs2)
    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio=0.8)
    return matches, locs1, locs2
