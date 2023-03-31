import cv2
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm


def sift_fc():
    img = cv2.imread('../data/cv_cover.jpg')
    test_img = cv2.imread('../data/cv_cover.jpg')
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    matches_list = []
    tmp = 1
    plt.suptitle('siftRotTest')
    for i in tqdm(range(36)):
        test_img_rot = scipy.ndimage.rotate(test_img, angle=i * 10)
        if (len(img.shape) >= 3):
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (len(test_img.shape) >= 3):
            test_img1 = cv2.cvtColor(test_img_rot, cv2.COLOR_BGR2GRAY)

        kp1, decs1 = sift.detectAndCompute(img1, None)
        kp2, decs2 = sift.detectAndCompute(test_img1, None)

        # matches = bf.knnMatch(decs1, decs2, k=2)
        # good_match = []
        # for m, n in matches:
        #     if m.distance < 0.75 * n.distance:
        #         good_match.append([m])
        # # print(len(good_match))
        # matches_list.append(len(good_match))
        matches = bf.match(decs1, decs2)
        matches_list.append(len(matches))
        if (i + 1) % 10 == 0:
            show_img = cv2.drawMatches(img, kp1, test_img1, kp2, matches, None)
            plt.subplot(3, 1, tmp)
            plt.imshow(show_img)
            tmp += 1
    plt.show()
    x = list(range(0, 360, 10))
    plt.suptitle('siftRotTest')
    plt.bar(x, matches_list, width=4)
    plt.show()


if __name__ == '__main__':
    sift_fc()
