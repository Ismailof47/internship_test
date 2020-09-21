import argparse
import os
import re
import json
from cnn_model import *
import torch
import torchvision.transforms as transforms
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
   
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='PyTorch CNN for faces classification')
	parser.add_argument('indir', type=str, help=' path to root dir with dir of images, same as ImageFolder ')
	args = parser.parse_args()

	model = Net()
	model.load_state_dict(torch.load('net_final.pt'))
	model.eval()

	data_dir = args.indir
	transform=transforms.Compose([transforms.Resize((227,227)),transforms.ToTensor(),])
	dataset = ImageFolderWithPaths(data_dir , transform) # our custom dataset
	
	data_out = {}
	batch_size = 64
	dataload = torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=4)
	class_idx = {0: "female" , 1: "male"}

	for batch_idx, (inputs, labels , path) in enumerate(dataload):
		outputs = model(inputs)
		ans = 	list(map(lambda x: class_idx[x] , torch.argmax(outputs,dim=-1).tolist()))
		name_out = list(map(lambda x : x[x.rfind('/')+1:] , (path) ))
		data_out.update( dict(zip(name_out, ans)))

	with open('data.txt', 'w') as outfile:
		json.dump(data_out, outfile)

