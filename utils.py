import cv2
from tqdm import tqdm
import uuid
import os


def __make_dir(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def video_to_image(video_file_path, main_image_folder="videos/images"):
	# convert video to Image format.
	cam = cv2.VideoCapture(video_file_path)
	file_number = 0
	image_paths = []
	id = uuid.uuid1()
	image_folder = "{}/{}".format(main_image_folder, int(id))
	__make_dir(image_folder)

	#print(image_folder)
	while True:
		ret, frame = cam.read()
		if ret: 
			image_path = "{}/filename_{}.png".format(image_folder, file_number)
			#print(image_path)
			cv2.imwrite(image_path, frame)
			file_number += 1
			image_paths.append(image_path)
		else:
			break
	return image_paths
	 