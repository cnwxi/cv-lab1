import os

import cv2
import numpy as np
from tqdm import tqdm

# Import necessary functions
from python.loadVid import loadVid
from python.planarH import compositeH


# Write script for Q4.1x
# opencv ORB
def Q4_1x():
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
    out = cv2.VideoWriter('../result/ar_orb.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, frame_size)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher()
    for i in tqdm(range(frames), desc='processing'):
        ar_frame = cv2.resize(ar[i, :, :, :], (cv_cover.shape[1], cv_cover.shape[0]))
        # ar_frame_gray = cv2.cvtColor(ar_frame, cv2.COLOR_BGR2GRAY)
        book_frame = book[i, :, :, :]
        # book_frame_gray = cv2.cvtColor(book_frame, cv2.COLOR_BGR2GRAY)
        # cv_cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = orb.detectAndCompute(book_frame, None)
        kp2, desc2 = orb.detectAndCompute(cv_cover, None)
        matches = bf.match(desc1, desc2)
        max_dis = matches[0].distance
        min_dis = matches[0].distance
        good = []
        for i in matches:
            min_dis = min(min_dis, i.distance)
            max_dis = max(max_dis, i.distance)
        # print('min', min_dis)
        for i in matches:
            if i.distance <= max(2 * min_dis, 30):
                good.append(i)
        matches = good
        if len(matches) < 4:
            continue
        img = cv2.drawMatches(cv_cover, kp1, book_frame, kp2, matches, None, flags=2)
        cv2.imshow('img', img)
        cv2.waitKey()

        locs1 = np.array([[kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1]] for i in matches])
        locs2 = np.array([[kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1]] for i in matches])

        bestH2to1, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC)
        comp_img = compositeH(bestH2to1, ar_frame, book_frame)
        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
        out.write(comp_img[:, :, ::-1])

    # plotMatches(img, test_img, matches, kp1, kp2)
    out.release()


if __name__ == '__main__':
    Q4_1x()
