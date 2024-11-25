GPU_NUM=3
SEQUENCES=("000" "001" "002" "003" "004" "005" "006" "007")

for SEQUENCE in ${SEQUENCES[@]}; do
    
        dataset=data/BSHuman/${SEQUENCE}

        CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py -s $dataset --eval \
        --motion_offset_flag --smpl_type smpl --actor_gender neutral \
        --exp_name BSHuman/${SEQUENCE} \
        --pose_mode "spline_4" \
        --model_blur_num 5 \
        --use_pose_offset \
        --use_lbs_offset \
        --pose_loss 1 

done
