import argparse
import os
import re
import json
from cnn_model import *
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='PyTorch CNN for faces classification')
	parser.add_argument('indir', type=str, help=' dir with image faces')
	args = parser.parse_args()

	model = Net()
	model.load_state_dict(torch.load('netb_final.pt'))
	model.eval()

	data_dir = args.indir
	imgs_name = os.listdir(data_dir)
	transform=transforms.Compose([transforms.Resize((227,227)),transforms.ToTensor(),])

	data_out = {}
	class_idx = {0: "female" , 1: "male"}

	for i in imgs_name:
		img = Image.open(data_dir + i , 'r')
		imt = transform(img)
		im = imt.unsqueeze(0)
		ans = model(im)
		data_out[i] =  class_idx[torch.argmax(ans).item()]

	with open('data.txt', 'w') as outfile:
		json.dump(data_out, outfile)

