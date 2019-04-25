import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Logger, read_json, write_json, save_checkpoint
import pickle

from train import train_model
from test import test_model


def print_keys():
	keys=h5_x.keys()
	for k in keys:
		for l in h5_x[k].keys():
			print (l)
		zz= h5_x[k]['user_summary']
		print(zz.shape)
		print(zz[0])
		break
		for i in (h5_x[k]['user_summary']):
			print("")
			print("")
			print("")
			for j in i:
				print(j)
		break


def information():
	keys=h5_x.keys()
	for k in keys:
		for l in h5_x[k].keys():
			print (l)
		for i in (h5_x[k]['change_points']):
			print (i)
		break


def main():
	h5_x = h5py.File('./eccv16_dataset_summe_google_pool5.h5','r')


	for hops in range(1,11):
		f_score=0.0
		for split in range(0,5):
			print("Current:",hops,split)
			model=train_model(split,hops)
			f_score+=test_model(i,model)
		print("Hops:",hops,",Fscore:",f_score/5)


main()