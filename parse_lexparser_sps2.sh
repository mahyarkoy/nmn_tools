#!/bin/sh
# Author info: Mahyar Khayatkhoei @ m.khayatkhoei@gmail.com
set -o errexit
READ_PATH=data/cvpr2016_cub/text_c10/*/*.txt
WRITE_PATH=data/sps2_none
PARSER_PATH=parser

counter=0
trap "exit" INT
for f in $READ_PATH
do
	b=${f##*/}
	b=${b%.*}
	d=${f%/*}
	dn=${d##*/}
	pd=${f%/*/*/*}
	fo=$WRITE_PATH/$dn/$b
	if ! [ -f "$fo.sps2" ];
	then
		mkdir -p $WRITE_PATH/$dn		
		$PARSER_PATH/lexparser.sh $f | python parse_sps2.py -f $fo || exit
		((counter++))
		echo ==========ITER: $counter
	fi
done || exit 1
