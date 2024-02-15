#!/bin/bash

for i in `seq 0 8`
do
	if [ ${i} -le 1 ];
	then
		CLASS=1
	elif [ ${i} -gt 1 -a ${i} -le 3 ];
	then
		CLASS=0
	elif [ ${i} -gt 3 -a ${i} -le 5 ];
	then
		CLASS=2
	else
		CLASS=3
	fi


	sed "s#DUMMY0#${CLASS}#g" gamma1e-3_level_sets_TERP.py > tmp.py
	sed -i "s#DUMMY1#${i}#g" tmp.py
	python tmp.py
	mkdir -v gamma1e-3_level_set_${i}
	mv -v class_${CLASS}/* gamma1e-3_level_set_${i}/
done
