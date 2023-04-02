for TASK in rte mrpc cola sst-2 qnli
do 
	bash scripts_large/train_${TASK}.sh
done
