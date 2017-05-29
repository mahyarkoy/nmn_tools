import cv2
import argparse
from os import walk
import os

# data blob size, input to the net
argprs = argparse.ArgumentParser()
argprs.add_argument("--impath", dest="imf", default="examples/images")
argprs.add_argument("--outpath", dest="outf", default="examples/images_aug")

def im_aug():
	args = argprs.parse_args()
	for (dirpath, dirnames, fnames) in walk(args.imf):
		for fn in fnames:
			impath = dirpath+'/'+fn
			im = cv2.imread(impath)
			imv = im.copy()
			imv = cv2.flip(imv,1)
			current_dir = args.outf+'/'+dirpath.split('/')[-1]
			if not os.path.exists(current_dir):
				os.system('mkdir -p '+current_dir)
			imoutdir = current_dir+'/'+fn.strip('.jpg')
			cv2.imwrite(imoutdir+'.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
			cv2.imwrite(imoutdir+'_fliph.jpg', imv, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == '__main__':
	im_aug()

	
