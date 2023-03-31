import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import necessary functions
from python.loadVid import loadVid
from python.planarH import compositeH


# Write script for Q4.1x
# opencv sift/orb
def Q4_1x(option):
    book = loadVid('../data/book.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    ar = loadVid('../data/ar_source.mov')
    if book is None:
        print('book is None')
        return
    if cv_cover is None:
        print('cv_cover is None')
        return
    if ar is None:
        print('ar is None')
        return
    ar = ar[:, 45:315, :, :]
    book_shape = np.shape(book)
    cover_shape = cv_cover.shape
    ar_shape = np.shape(ar)
    centerx = ar_shape[2] // 2
    x = centerx - cover_shape[1] / 2
    ar = ar[:, :, int(x):int(x + cover_shape[1]), :]
    ar_shape = np.shape(ar)
    frames = min(ar_shape[0], book_shape[0])
    frame_size = (book_shape[2], book_shape[1])
    folder = os.path.exists('../result')
    if not folder:
        os.makedirs('../result')
    sift = cv2.SIFT_create()
    if option == 'sift':
        detect_fc = cv2.SIFT_create()
        path = '../result/ar_sift.avi'
    elif option == 'orb':
        detect_fc = cv2.BFMatcher()
        path = '../result/ar_orb.avi'
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, frame_size)

    plt.ion()
    for i in tqdm(range(frames), desc='processing'):
        ar_frame = cv2.resize(ar[i, :, :, :], (cv_cover.shape[1], cv_cover.shape[0]))
        # ar_frame_gray = cv2.cvtColor(ar_frame, cv2.COLOR_BGR2GRAY)
        book_frame = book[i, :, :, :]
        book_frame_gray = cv2.cvtColor(book_frame, cv2.COLOR_BGR2GRAY)
        cv_cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = sift.detectAndCompute(cv_cover_gray, None)
        kp2, desc2 = sift.detectAndCompute(book_frame_gray, None)
        matches = detect_fc.knnMatch(desc1, desc2, k=2)
        good_match = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_match.append([m])

        matches = [i[0] for i in good_match]
        if len(matches) < 4:
            continue
        img = cv2.drawMatchesKnn(cv_cover, kp1, book_frame, kp2, good_match, None, flags=2)
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        locs1 = np.array([[kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1]] for i in matches])
        locs2 = np.array([[kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1]] for i in matches])

        bestH2to1, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC)
        comp_img = compositeH(bestH2to1, ar_frame, book_frame)
        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
        out.write(comp_img[:, :, ::-1])
        plt.subplot(1, 2, 2)
        plt.imshow(comp_img)
        plt.pause(0.01)
        plt.clf()
    # plotMatches(img, test_img, matches, kp1, kp2)
    out.release()


if __name__ == '__main__':
    Q4_1x('orb')
