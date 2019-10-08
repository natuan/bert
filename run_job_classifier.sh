#!/bin/bash

sessions=OCT_04
drop_rates=(0.1)
l2_scales=(0)
learning_rates=(3e-05)
label_smoothings=(0)
epochs=(600)

for sess in ${sessions[*]}; do
for drop in ${drop_rates[*]}; do
for l2 in ${l2_scales[*]}; do
for ls in ${label_smoothings[*]}; do
for lr in ${learning_rates[*]}; do
for ep in ${epochs[*]}; do
    python job_classifier.py \
	   --session=$sess \
	   --output_drop_rate=$drop \
	   --l2_scale=$l2 \
	   --label_smoothing=$ls \
	   --learning_rate=$lr \
	   --num_train_epochs=$ep \
	   --do_train=True
done
done
done
done
done
done

