import cv2
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm


def rot_fc(option):
    img = cv2.imread('../data/cv_cover.jpg')
    test_img = cv2.imread('../data/cv_cover.jpg')
    if option == 'sift':
        detect_fc = cv2.SIFT_create()
    else:
        detect_fc = cv2.ORB_create()
    bf = cv2.BFMatcher()
    matches_list = []
    tmp = 1
    plt.suptitle(f'{option}RotTest')
    for i in tqdm(range(36)):
        test_img_rot = scipy.ndimage.rotate(test_img, angle=i * 10)
        if (len(img.shape) >= 3):
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (len(test_img.shape) >= 3):
            test_img1 = cv2.cvtColor(test_img_rot, cv2.COLOR_BGR2GRAY)

        kp1, decs1 = detect_fc.detectAndCompute(img1, None)
        kp2, decs2 = detect_fc.detectAndCompute(test_img1, None)

        matches = bf.match(decs1, decs2)
        matches_list.append(len(matches))
        if (i + 1) % 10 == 0:
            show_img = cv2.drawMatches(img, kp1, test_img1, kp2, matches, None)
            plt.subplot(3, 1, tmp)
            plt.imshow(show_img)
            tmp += 1
    plt.show()
    x = list(range(0, 360, 10))
    plt.suptitle(f'{option}RotTest')
    plt.bar(x, matches_list, width=4)
    plt.show()


if __name__ == '__main__':
    for i in ['sift', 'orb']:
        rot_fc(i)
