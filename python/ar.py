import os

import cv2
import numpy as np
from tqdm import tqdm

# Import necessary functions
from loadVid import loadVid
from matchPics import matchPics
from planarH import compositeH
from planarH import computeH_ransac


# Write script for Q4.1
def Q4_1():
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
    out = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, frame_size)
    for i in tqdm(range(frames), desc='processing'):
        ar_frame = cv2.resize(ar[i, :, :, :], (cv_cover.shape[1], cv_cover.shape[0]))
        # ar_frame_gray = cv2.cvtColor(ar_frame, cv2.COLOR_BGR2GRAY)
        book_frame = book[i, :, :, :]
        # book_frame_gray = cv2.cvtColor(book_frame, cv2.COLOR_BGR2GRAY)
        # cv_cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)

        matches, locs1, locs2 = matchPics(book_frame, cv_cover)
        locs1 = locs1[matches[:, 0]]
        locs1 = locs1[:, [1, 0]]
        locs2 = locs2[matches[:, 1]]
        locs2 = locs2[:, [1, 0]]
        bestH2to1, inliers = computeH_ransac(locs1, locs2)
        comp_img = compositeH(bestH2to1, ar_frame, book_frame)
        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
        out.write(comp_img[:, :, ::-1])

    out.release()


if __name__ == '__main__':
    Q4_1()
