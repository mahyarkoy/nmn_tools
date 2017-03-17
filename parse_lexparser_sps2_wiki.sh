#!/bin/sh
set -o errexit
READ_PATH=/media/evl/Public/Mahyar/Data/Columbia/text_c10/*/*.txt
WRITE_PATH=/media/evl/Public/Mahyar/Data/Columbia/Downloads/CVPRdata/sps2
counter=0
trap "exit" INT
for f in $READ_PATH
do
	b=${f##*/}
	b=${b%.*}
	d=${f%/*}
	dn=${d##*/}
	pd=${f%/*/*/*}
	fo=$pd/sps2/$dn/$b
	if ! [ -f "$fo.sps2" ];
	then
		mkdir -p $pd/sps2/$dn		
		./lexparser.sh $f | python ~/parse_sps2.py -f $fo || exit
		((counter++))
		echo ==========ITER: $counter
	fi
done || exit 1
