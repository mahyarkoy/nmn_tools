lexparser_copy.sh
------
from stanford dependency parser.

parse_lexparser_sps2.sh (aka parse_all)
------
runs lexparser on sentences of each description file, feeds it to parse_sps2.py to save as sps2 format.

parse_sps2.py
------
parses a given line into a set of sps2 phrases: (is (and beak black))

create_db_cub.py
------
generates batches for training, evaluating and testing for nmn model, from cub data set and sps2 parses.

train_test_split.mat
------
split from Mohamed on cub.

zero_shot_calc.py
------
evaluates the task of zeroshot learning from nmn results.

pred_anslysis.py
------
calculates statistics based on a nmn logs json file.

spell_check.py
------
spell checking script found online.

big.txt
------
vocabulary used by spell_check.

ex_generator.py
------
code for generating negative examples (from cluster_info.pk) by olivia. (modified for minor fixes)

check_sps2.py
------
goes over the files parsed from sentence to sps2 format, and provides some statistics on the generated parse data set.

extract_vgg_cv.py
------
uses caffe to run vgg 16 forward pass on images and extract pool 5 layer features. has batching and uses opencv.

im_aug.py
------
horizontally flips images on the given path, and save alongside original images in a given output path.

