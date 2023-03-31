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
def load():
    book = loadVid('../data/book.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    ar = loadVid('../data/ar_source.mov')

    return book, cv_cover, ar


def Q4_1x(option, book, cv_cover, ar):
    # book = loadVid('../data/book.mov')
    # cv_cover = cv2.imread('../data/cv_cover.jpg')
    # ar = loadVid('../data/ar_source.mov')
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
    elif option == 'orb':
        detect_fc = cv2.ORB_create()
    path = f'../result/ar_{option}.avi'
    bf = cv2.BFMatcher()
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, frame_size)

    plt.ion()

    for i in tqdm(range(frames), desc='processing'):
        ar_frame = cv2.resize(ar[i, :, :, :], (cv_cover.shape[1], cv_cover.shape[0]))
        # ar_frame_gray = cv2.cvtColor(ar_frame, cv2.COLOR_BGR2GRAY)
        book_frame = book[i, :, :, :]
        book_frame_gray = cv2.cvtColor(book_frame, cv2.COLOR_BGR2GRAY)
        cv_cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = detect_fc.detectAndCompute(cv_cover_gray, None)
        kp2, desc2 = detect_fc.detectAndCompute(book_frame_gray, None)

        good_match = []
        matches = bf.knnMatch(desc1, desc2, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_match.append([m])

        matches = [i[0] for i in good_match]
        if len(matches) < 4:
            continue

        img = cv2.drawMatchesKnn(cv_cover, kp1, book_frame, kp2, good_match, None, flags=2)
        plt.suptitle(f'{option}')
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        locs1 = np.array([[kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1]] for i in matches])
        locs2 = np.array([[kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1]] for i in matches])

        bestH2to1, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC, 5.0)
        comp_img = compositeH(bestH2to1, ar_frame, book_frame)
        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
        out.write(comp_img[:, :, ::-1])

        plt.subplot(1, 2, 2)
        plt.imshow(comp_img)
        plt.pause(0.01)
        plt.clf()
    out.release()


if __name__ == '__main__':
    book, cv_cover, ar = load()
    for i in ['sift', 'orb']:
        Q4_1x(i, book, cv_cover, ar)
