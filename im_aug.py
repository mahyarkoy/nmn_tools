import cv2
import argparse
from os import walk
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import numpy as np

# data blob size, input to the net
argprs = argparse.ArgumentParser()
argprs.add_argument("--impath", dest="imf", default="examples/images")
argprs.add_argument("--outpath", dest="outf", default="examples/images_aug_bb")
argprs.add_argument("--bbox", dest="bbf", default=None)
argprs.add_argument("--fnames", dest="fnf", default=None)

def im_aug():
	args = argprs.parse_args()
	print('>>> Initializing image augmentation')
	bbox_dict = dict()
	if args.bbf and args.fnf:
		print('>>> Reading bounding boxes')
		bbox_dict = load_bbox(args.bbf, args.fnf)
	dir_count = 0
	for (dirpath, dirnames, fnames) in walk(args.imf):
		if len(fnames) == 0:
			continue
		print('>>> At directory: %s' % dirpath)
		widgets = ["directory #%d|" % (dir_count), Percentage(), Bar(), ETA()]
		pbar = ProgressBar(maxval=len(fnames), widgets=widgets)
		pbar.start()
		dir_count += 1
		fn_count = 0
		for fn in fnames:
			impath = dirpath+'/'+fn
			im = cv2.imread(impath)
			### crop the bbox from the image if it exists
			if fn in bbox_dict.keys():
				im = custom_crop(im, bbox_dict[fn])
			imv = im.copy()
			imv = cv2.flip(imv,1)
			current_dir = args.outf+'/'+dirpath.split('/')[-1]
			if not os.path.exists(current_dir):
				os.system('mkdir -p '+current_dir)
			imoutdir = current_dir+'/'+fn.strip('.jpg')
			cv2.imwrite(imoutdir+'.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
			cv2.imwrite(imoutdir+'_fliph.jpg', imv, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
			fn_count += 1
			pbar.update(fn_count)

def load_bbox(bboxs_path, filenames_path):
	bbox_dict = dict()
	with open(bboxs_path, 'r') as bfs, open(filenames_path, 'r') as nfs:
	    ### use filename (no folder prefix) as keys, and bbox list as values
	    for l, n in zip(bfs, nfs):
	        bbox_dict[n.strip().split('/')[-1]] = [float(str) for str in l.strip().split(' ')[1:]]
	return bbox_dict

def custom_crop(img, bbox):
    # bbox = [x-left, y-top, width, height]
    imsiz = img.shape  # [height, width, channel]
    # if box[0] + box[2] >= imsiz[1] or\
    #     box[1] + box[3] >= imsiz[0] or\
    #     box[0] <= 0 or\
    #     box[1] <= 0:
    #     box[0] = np.maximum(0, box[0])
    #     box[1] = np.maximum(0, box[1])
    #     box[2] = np.minimum(imsiz[1] - box[0] - 1, box[2])
    #     box[3] = np.minimum(imsiz[0] - box[1] - 1, box[3])
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    y1 = np.maximum(0, center_y - R)
    y2 = np.minimum(imsiz[0], center_y + R)
    x1 = np.maximum(0, center_x - R)
    x2 = np.minimum(imsiz[1], center_x + R)
    img_cropped = img[y1:y2, x1:x2, :]
    return img_cropped

if __name__ == '__main__':
	im_aug()

	
