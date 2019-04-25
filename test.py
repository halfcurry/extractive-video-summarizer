import numpy as np
import h5py

import knapsack as knapsack_dp

def test_model(split_id,model):

	db = "summe" #db to test

	h5_x = h5py.File('./eccv16_dataset_'+db+'_google_pool5.h5','r')

	shots = pickle.load(open("./shots/"+db+"_shots.pkl", "rb"))
	shots_gt = pickle.load(open("./shots/"+db+"_shots_gt.pkl", "rb"))

	splits = read_json(db+"_splits.json")
	# print(splits)

#     split_id = 1 #split to test

	split = splits[split_id]
	train_keys = split['train_keys']
	test_keys = split['test_keys']

	tidxs = np.arange(len(test_keys))

	model.eval()
	avg_f=0.0
	# test_avg_loss=0
	for idx in tidxs:
		key = test_keys[idx]
#         print(h5_x[key]['video_name'].value)
#         print(key)
		seq = shots[key]
		seq = torch.from_numpy(seq).unsqueeze(0)
		seq_gt = shots_gt[key].reshape((300,1))
	#     print(seq_gt)
		seq_gt = torch.from_numpy(seq_gt).unsqueeze(0)

	#         print(seq.shape)
	#         print(seq_gt.shape)
		seq.cuda()
		seq_gt.cuda()

	#         print(seq.type())
		out = model(seq.float())

		segments = h5_x[key]['change_points']
		num_frames = int(h5_x[key]['n_frames'].value)
	#     print(type(num_frames))
	#     print(len(segments))
	#     print(segments[0])
	#     print(out)

	#     tvalues = []
		values = []
	#     print(segments[1])
		for i in range(len(segments)):
			values.append(float(out[0][i]))
	#         for j in range(segments[i][1]-segments[i][0]+1):
	#             tvalues.append(float(out[0][i]))
	#     print(float(out[0][0]))
	#     print(values)
	#     print(len(values))

	#     assert(num_frames==len(values))
	#     print(num_frames)
		weights = h5_x[key]['n_frame_per_seg'][...].tolist()
	#     print(type(weights))
	#     weights = [1 for i in range(num_frames)]
		n_segs = len(segments)
		capacity = int(round(0.15 * num_frames))
	#     capacity = round(capacity)
	#     print(len(values))
	#     print(len(weights))
	#     print(num_frames)
	#     print(capacity)
	#     print(values)
	#     print(weights)

		selected = knapsack_dp(values, weights, n_segs, capacity)

	#     print(capacity)
	#     print(selected)
	#     print(picks)

		summary = np.zeros((1), dtype=np.float32) # this element should be deleted
		for seg_idx in range(n_segs):
			nf = weights[seg_idx]
			if seg_idx in selected:
				tmp = np.ones((nf), dtype=np.float32)
			else:
				tmp = np.zeros((nf), dtype=np.float32)
			summary = np.concatenate((summary, tmp))

		summary = np.delete(summary, 0)
	#     print (summary)
		gt_summary= h5_x[key]['user_summary'][...]
		f,_,_= evaluate_summary(summary,gt_summary,'max')
#         print ("fscore:",f)
		avg_f+=f
	#     break

	return (avg_f/len(tidxs))