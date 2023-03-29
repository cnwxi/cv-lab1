import cv2
import numpy as np
from tqdm import tqdm


def loadVid(path):
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	cap = cv2.VideoCapture(path)

	# Check if camera opened successfully
	if (cap.isOpened() == False):
		print("Error opening video stream or file")
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	i = 0
	# Read until video is completed
	with tqdm(total=frame_count) as pbar:
		pbar.set_description('video loading')
		while (cap.isOpened()):
			# Capture frame-by-frame
			i += 1
			ret, frame = cap.read()
			if ret == True:
				# Store the resulting frame
				if i == 1:
					frames = frame[np.newaxis, ...]
				else:
					frame = frame[np.newaxis, ...]
					frames = np.vstack([frames, frame])
					frames = np.squeeze(frames)
				pbar.update(1)
			else:
				break

	# When everything done, release the video capture object
	cap.release()

	return frames
