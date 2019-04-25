import numpy as np
import pickle

def train_model(split_id,n_hops=1):
#     print("Hops:",n_hops)

	db = "summe"

	shots = pickle.load(open("./shots/"+db+"_shots.pkl", "rb"))
	shots_gt = pickle.load(open("./shots/"+db+"_shots_gt.pkl", "rb"))

	splits = read_json(db+"_splits.json")
	# print(splits)
	# split_id = 1

	split = splits[split_id]
	train_keys = split['train_keys']
	test_keys = split['test_keys']


	# optimizer, model load and stuff
	#Declaring the model
	model = MA_module_multiple(n_hops)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
	loss_criterion=torch.nn.MSELoss()

	#for all train indices
	tidxs = np.arange(len(test_keys))
	num_epochs = 100
	for epoch in range(num_epochs):
		idxs = np.arange(len(train_keys))
		np.random.shuffle(idxs)


#         if epoch%20==0:
#             print (epoch)

		model.train()
		avg_loss=0
		for idx in idxs:
			key = train_keys[idx]
			seq = shots[key]
			seq = torch.from_numpy(seq).unsqueeze(0)
			seq_gt = shots_gt[key].reshape((300,1))
			seq_gt = torch.from_numpy(seq_gt).unsqueeze(0)

	#         print(seq.shape)
	#         print(seq_gt.shape)
			seq.cuda()
			seq_gt.cuda()

	#         print(seq.type())
			out = model(seq.float())
			cost = loss_criterion(out,seq_gt.float())
			avg_loss+=cost.item()
	#         print (cost.item())
			optimizer.zero_grad()
			cost.backward()
			optimizer.step()

		#calculation of test loss
		model.eval()
		test_avg_loss=0
		for idx in tidxs:
			key = test_keys[idx]
			seq = shots[key]
			seq = torch.from_numpy(seq).unsqueeze(0)
			seq_gt = shots_gt[key].reshape((300,1))
			seq_gt = torch.from_numpy(seq_gt).unsqueeze(0)

	#         print(seq.shape)
	#         print(seq_gt.shape)
			seq.cuda()
			seq_gt.cuda()

	#         print(seq.type())
			out = model(seq.float())
	#         print(out)
			cost = loss_criterion(out,seq_gt.float())
			test_avg_loss+=cost.item()
	#         print (cost.item())

#         print("train_loss",avg_loss/20.0,"test_loss",test_avg_loss/5.0)
	print("Trained")
	return model