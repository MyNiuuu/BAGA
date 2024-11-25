GPU_NUM=2
DATA_BLUR_NUMS=(5 7 9 11)
SEQUENCES=("my_377" "my_386" "my_387" "my_392" "my_393" "my_394")


for DATA_BLUR_NUM in ${DATA_BLUR_NUMS[@]}; do

    MODEL_BLUR_NUM=$DATA_BLUR_NUM

    for SEQUENCE in ${SEQUENCES[@]}; do

        dataset=./data/BlurZJU/blur${DATA_BLUR_NUM}/${SEQUENCE}

        CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py -s $dataset --eval \
        --motion_offset_flag --smpl_type smpl --actor_gender neutral \
        --exp_name BlurZJU/dblur${DATA_BLUR_NUM}/${SEQUENCE} \
        --pose_mode "spline_4" \
        --model_blur_num $MODEL_BLUR_NUM \
        --data_blur_num $DATA_BLUR_NUM \
        --use_pose_offset \
        --use_lbs_offset \
        --pose_loss 1 \

    done

done