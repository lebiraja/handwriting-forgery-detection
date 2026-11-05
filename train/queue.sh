#!/bin/bash
cd /home/lebi/Projects/lincy/final-year/train || exit 1
W="num_workers=12"
./run_train.sh main_s1 --set seed=1 $W
./run_train.sh main_s2 --set seed=2 $W
S="epochs=40 warmup_epochs=2 patience=40 $W"
./run_train.sh abl_main40      --set $S
./run_train.sh abl_no_distill  --set $S distill.enabled=false
./run_train.sh abl_mean_pool   --set $S pool=mean
./run_train.sh abl_no_cmatch   --set $S content_matched=false
./run_train.sh abl_no_mining   --set $S p_mine=0
./run_train.sh abl_k8          --set $S k=8
./run_train.sh abl_no_aug      --set $S augment=false
echo QUEUE_DONE > runs/queue.done
