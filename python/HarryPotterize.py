import cv2
from matplotlib import pyplot as plt

# Import necessary functions
from matchPics import matchPics
from planarH import compositeH
from planarH import computeH_ransac


# Write script for Q3.9
def Q3_9():
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    matches, locs1, locs2 = matchPics(cv_desk, cv_cover)
    locs1 = locs1[matches[:, 0]]
    locs1 = locs1[:, [1, 0]]
    locs2 = locs2[matches[:, 1]]
    locs2 = locs2[:, [1, 0]]
    bestH2to1, inliers = computeH_ransac(locs1, locs2)
    print(bestH2to1)
    hp_cover = cv2.resize(hp_cover, ((cv_cover.shape[1]), cv_cover.shape[0]))
    comp_img = compositeH(bestH2to1, hp_cover, cv_desk)
    cv2.imwrite('../data/composite_img.jpg', comp_img)
    list = [cv_cover, cv_desk, hp_cover, comp_img]
    plt.figure()
    for i, item in enumerate(list):
        plt.subplot(1, 4, i + 1)
        plt.imshow(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    Q3_9()
