import cv2
import matplotlib.pyplot as plt
import numpy as np


def panaroma(option):
    # img1 = cv2.imread('../data/right.png')
    # img2 = cv2.imread('../data/left.png')
    img1 = cv2.imread('../data/pano_right.jpg')
    img2 = cv2.imread('../data/pano_left.jpg')
    img1_gray = img1
    img2_gray = img2

    if len(img1_gray.shape) >= 3:
        img1_gray = cv2.cvtColor(img1_gray, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) >= 3:
        img2_gray = cv2.cvtColor(img2_gray, cv2.COLOR_BGR2GRAY)
    if option == 'sift':
        detect_fc = cv2.SIFT_create()
    elif option == 'orb':
        detect_fc = cv2.ORB_create()
    else:
        return
    bf = cv2.BFMatcher()
    kp1, desc1 = detect_fc.detectAndCompute(img1_gray, None)
    kp2, desc2 = detect_fc.detectAndCompute(img2_gray, None)
    good_match = []
    matches = bf.knnMatch(desc1, desc2, k=2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_match.append([m])
    matches = [i[0] for i in good_match]
    img_show = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_match, None, flags=2)

    locs1 = np.array([kp1[i.queryIdx].pt for i in matches])
    locs2 = np.array([kp2[i.trainIdx].pt for i in matches])
    bestH2to1, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC, 4.0)
    r, c = int(2 * img2.shape[1]), int(1.5 * img2.shape[0])

    output = cv2.warpPerspective(img1, bestH2to1, (r, c))
    tmp = output.copy()
    output[0:img2.shape[0], 0:img2.shape[1]] = img2
    plt.suptitle(f'{option}')
    for i, element in enumerate([img_show, output]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(cv2.cvtColor(element, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(f'../data/pano_out_{option}.jpg', output)


if __name__ == '__main__':
    for i in ['orb', 'sift']:
        panaroma(i)
