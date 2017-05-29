#!/usr/bin/env python2

import sys
import argparse
from os import walk
caffe_root = ""
sys.path.insert(0, caffe_root + "python")

import caffe
import numpy as np
import matplotlib.pyplot as plt
import cv2

# data blob size, input to the net
argprs = argparse.ArgumentParser()
argprs.add_argument("--impath", dest="imFolder", default="examples/images")
argprs.add_argument("--outpath", dest="outFolder", default="examples/conv")
argprs.add_argument("--meanpath", dest="meanFile", default="python/caffe/imagenet/ilsvrc_2012_mean.npy")
img_size = 448
batch_size = 10

def extract_feature():
	print "hello"
	args = argprs.parse_args()
	imf = args.imFolder
	outf = args.outFolder
	meanfile = args.meanFile
	## load model and weights for 16 level VGGNet
	modelDef = "models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt"
	modelWeights = "models/vgg/VGG_ILSVRC_16_layers.caffemodel"
	#modelDef = 'models/bvlc_reference_caffenet/deploy.prototxt'
	#modelWeights = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

	net, trans = setup_caffe_net(modelDef, modelWeights, meanfile)
	counter = 0
	#fnames = list()
	for (dirpath, dirnames, fnames) in walk(imf):
		for i in xrange(0,len(fnames),batch_size):
			fchunk = fnames[i:i+batch_size]
			idx = 0
			for fn in fchunk:
				impath = dirpath+'/'+fn
				imtrans = preproc_img(impath, img_size, trans)
				#print imtrans.shape
				net.blobs["data"].data[idx,...] = np.copy(imtrans)
				idx += 1
			net.forward()
			idx = 0
			for fn in fchunk:
				conv5 = net.blobs["pool5"].data[idx,...]
				outfname = outf+'/'+fn
				np.savez(outfname, conv5)
				idx += 1
				print counter
				counter += 1
"""
		for fn in filesnames:
			print counter
			counter +=1
			impath = imf+fn
			imtrans = preproc_img(impath, img_size, trans)
			print imtrans.shape
			net.blobs["data"].data[0,...] = imtrans
			net.forward()
			conv5 = net.blobs["pool5"].data
			print conv5.shape
			outfname = outf+fn
			np.savez(outfname, conv5[0,...])
"""
def resize_crop_image(input_file, output_side_length = 256):
        '''Takes an image name, resize it and crop the center square
        '''
        img = cv2.imread(input_file)
        height, width, depth = img.shape
        new_height = output_side_length
        new_width = output_side_length
        #if height > width:
        #    new_height = output_side_length * height / width
        #else:
        #    new_width = output_side_length * width / height
        resized_img = cv2.resize(img, (new_width, new_height))
        #height_offset = (new_height - output_side_length) / 2
        #width_offset = (new_width - output_side_length) / 2
        #cropped_img = resized_img[height_offset:height_offset + output_side_length,
        #                          width_offset:width_offset + output_side_length]
        cropped_img = resized_img
        return cropped_img


def setup_caffe_net(modelDef, modelWeights, meanfile):
	## setup the net
	caffe.set_mode_gpu();
	caffe.set_device(2);
	#caffe.set_mode_cpu();
	net = caffe.Net(modelDef, modelWeights, caffe.TEST)
	## reshape net
	net.blobs["data"].reshape(batch_size, 3, img_size, img_size)
	net.reshape();
	#print "Blobs in/out"
	#print net.blobs["data"].data.shape
	#print net.blobs["prob"].data.shape
	## calculate images mean
	mu = np.load(meanfile)
	mum = mu.mean(1).mean(1)
	#print "mean-subtracted values:", zip("BGR", mum)

	## set transformer: HWC to CHW, deduct mean, RGB to BGR
	trans = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
	trans.set_transpose("data", (2,0,1))
	trans.set_mean("data", mum)
	trans.set_raw_scale("data", 255)
	#trans.set_channel_swap("data", (2,1,0))
	return net, trans
	
def preproc_img(imgPath, imgSize, trans):
	img = resize_crop_image(imgPath, imgSize)
	transImg = trans.preprocess("data", img)
	return transImg

def display_labels(labPath, net):
	probs = net.blobs["prob"].data[0,:]

	## load labels
	labsFile = "data/ilsvrc12/synset_words.txt"
	labs = np.loadtxt(labsFile, str, delimiter="\n")

	## get top 5 labels with probabilities
	topIdx = probs.argsort()[::-1][:5] # extended slicing
	print "probs and labels:"
	print zip(list(probs[topIdx]), list(labs[topIdx]))

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

if __name__ == "__main__":
    extract_feature()