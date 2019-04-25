import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse
import pickle
import cv2

# Opens the Video file

def save_video_frames(frame_dir, video_name):
	if not osp.exists(frame_dir):
		os.mkdir(frame_dir) 
	cap= cv2.VideoCapture(video_name)
	width = None
	height = None
	i=1

	ret = True

	while ret:
		print(i)
		ret, frame = cap.read()
		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		cv2.imwrite(frame_dir + str(i).zfill(6) + '.jpg',frame)
		i+=1 
	cap.release()
	cv2.destroyAllWindows()

	return width, height




# h5_res = h5py.File(osp.join(save_dir, 'result.h5'), 'w')

# h5_res.create_dataset(video_name + '/machine_summary', data=machine_summary)

def frm2video(frm_dir, summary, vid_writer, width, height):
	for idx, val in enumerate(summary):
		if val == 1:
			# here frame name starts with '000001.jpg'
			# change according to your need
			frm_name = str(idx+1).zfill(6) + '.jpg'
			frm_path = osp.join(frm_dir, frm_name)
			frm = cv2.imread(frm_path)
			frm = cv2.resize(frm, (width, height))
			vid_writer.write(frm)


def create_summary(path, frame_dir, video_name, save_dir, save_name, fps):
	# width, height = save_video_frames(frame_dir, video_name)
	# print("video frames done")

	if not osp.exists(save_dir):
		os.mkdir(save_dir)

	vid_writer = cv2.VideoWriter(
		osp.join(save_dir, save_name),
		cv2.VideoWriter_fourcc(*'MP4V'),
		fps,
		(1920, 1080),
	)

	# h5_res = h5py.File(args.path, 'r')
	# key = h5_res.keys()[args.idx]
	# summary = h5_res[key]['machine_summary'][...]
	# h5_res.close()
	print(path)
	summary = pickle.load(open(path, "rb"))[8]
	print(summary)
	frm2video(frame_dir, summary, vid_writer, 1920, 1080)
	vid_writer.release()
	

path = "./Notre_Dame_gt.pkl"
frame_dir = "./frames/"
video_name = "./Notre_Dame.mp4"
save_dir = "./gen_summary/"
save_name = "Notre_summary_gt.mp4"
fps = 30

create_summary(path, frame_dir, video_name, save_dir, save_name, fps)
