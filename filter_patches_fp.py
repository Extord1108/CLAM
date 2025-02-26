import time
import os
import argparse
import pdb
from functools import partial
import multiprocessing

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import cv2

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP

class Dataset_All_Bags_Filter(Dataset_All_Bags):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		seg_params = {
			'sthresh': self.df['sthresh'][idx],
			'mthresh': self.df['mthresh'][idx],
			'close': self.df['close'][idx],
			'use_otsu': self.df['use_otsu'][idx],
		}

		return self.df['slide_id'][idx], seg_params



def is_patch_keep(img,tissue_ratio,**kwargs):
	# 根据前景占比判断是否保留

	sthresh = kwargs['sthresh']
	mthresh = kwargs['mthresh']
	close = kwargs['close']
	use_otsu = kwargs['use_otsu']

	img = np.array(img)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_med = cv2.medianBlur(img_hsv[:,:,1], kwargs['mthresh'])

	if use_otsu:
		_, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	else:
		_,img_otsu = cv2.threshold(img_med, kwargs['sthresh'], 255, cv2.THRESH_BINARY)
	
	if close > 0:
		kernel = np.ones((close, close), np.uint8)
		img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

	contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	mask = np.zeros_like(img_otsu)
	cv2.drawContours(mask, contours, -1, (1,1,1), thickness=cv2.FILLED)

	foreground_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])

	return foreground_ratio > tissue_ratio

	

def init_worker(slide_file_path):
	global worker_slide
	worker_slide = openslide.open_slide(slide_file_path)

def worker_func(coord_input):
	global worker_slide
	coord = coord_input["coord"]
	patch_size = coord_input["patch_size"]
	tissue_ratio = coord_input["tissue_ratio"]
	seg_params = coord_input["seg_params"]
	patch = worker_slide.read_region(coord, 0, (patch_size, patch_size)).convert('RGB')

	if is_patch_keep(patch,tissue_ratio, **seg_params):
		return coord
	else:
		return None
    

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--tissue_ratio', type=float, default=0.4)
args = parser.parse_args()


if __name__ == '__main__':
	print('filter patches depending on the ratio of tissue')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags_Filter(csv_path)
	
	output_path = os.path.join(args.save_dir, 'filtered_patches')

	os.makedirs(args.save_dir, exist_ok=True)
	os.makedirs(output_path, exist_ok=True)

	total = len(bags_dataset)

	for bag_candidate_idx in tqdm(range(total)):
		slide_id,seg_params = bags_dataset[bag_candidate_idx]
		slide_id = slide_id.split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		if os.path.exists(os.path.join(output_path,bag_name)):
			continue
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		time_start = time.time()

		hdf5_file = h5py.File(h5_file_path, 'r')
		patch_size = hdf5_file['coords'].attrs['patch_size']
		coords_output = []
		coords_input = []
		for coord in hdf5_file['coords']:
			coords_input.append({
				'coord': coord,
				'patch_size': patch_size,
				'tissue_ratio': args.tissue_ratio,
				'seg_params': seg_params
			})

		with multiprocessing.Pool(
			processes=16, 
			initializer=init_worker,
			initargs=(slide_file_path,)
		) as pool:
			coords_output = pool.map(worker_func,coords_input)

		time_elapsed = time.time() - time_start
		print('\nfilter patches for {} took {} s'.format(slide_id, time_elapsed))
		coords_output = [coord for coord in coords_output if coord is not None]
		coords = np.array(coords_output)
		print('original patches num:{}       filtered patches num:{}'.format(len(coords_input), len(coords_output)))


		
		asset_dict = {'coords' :          coords}
		attr = hdf5_file['coords'].attrs
		attr_dict = { 'coords' : attr}
		save_hdf5(os.path.join(output_path,bag_name), asset_dict, attr_dict, mode='w')




