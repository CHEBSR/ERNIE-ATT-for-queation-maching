# unset CUDA_VISIBLE_DEVICES
# python -u -m paddle.distributed.launch --gpus "0,1" train.py \
#        --train_set train_all.txt \
#        --dev_set dev.txt \
#        --device gpu \
#        --eval_step 500 \
#        --save_dir ./checkpoints \
#        --train_batch_size 24 \
#        # --learning_rate 2E-5 \
#        --learning_rate 5E-6 \
#        --rdrop_coef 0.1 \
#        --plm_name "ernie-3.0-xbase-zh" \
#        --save_step 1000

# ernie-3.0-xbase-zh "roberta-wwm-ext-large" ernie-3.0-medium-zh ernie-gram-zh

# python train.py --train_set train_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints --train_batch_size 24 --learning_rate 5E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set dev.txt --save_step 1000 --init_from_ckpt ./checkpoints/model_1000/model_state.pdparams --max_seq_length 64
# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set train_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints --train_batch_size 24 --learning_rate 5E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set dev.txt --save_step 1000 --max_seq_length 64  --init_from_ckpt ./checkpoints/model_1000/model_state.pdparams
# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set train_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints_xbase-zh --train_batch_size 16 --learning_rate 5E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set dev.txt --save_step 1000 --max_seq_length 64  --init_from_ckpt ./checkpoints/model_3000/model_state.pdparams

# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set train.txt --device gpu --eval_step 500 --save_dir ./checkpoints_xbase-zh --train_batch_size 32 --learning_rate 5E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set dev.txt --save_step 2000 --max_seq_length 128

# python predict.py --device gpu --params_path checkpoints_xbase-zh/model_44000/model_state.pdparams --plm_name ernie-3.0-xbase-zh

# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set train_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints_base-zh --train_batch_size 64 --learning_rate 5E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-base-zh" --dev_set dev.txt --save_step 1000 --max_seq_length 64

# python -u -m paddle.distributed.launch --gpus "0" train.py --train_set train_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints_medium-zh --train_batch_size 64 --learning_rate 5E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-medium-zh" --dev_set dev.txt --save_step 1000 --max_seq_length 64

# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set train_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints_ernie-gram-zh --train_batch_size 64 --learning_rate 2E-5 --rdrop_coef 0.1 --plm_name "ernie-gram-zh" --dev_set dev.txt --save_step 1000 --max_seq_length 64
# python predict.py --device gpu --params_path "./checkpoints_ernie-gram-zh/model_20000/model_state.pdparams" --batch_size 32 --max_seq_length 64  --plm_name "ernie-gram-zh" --input_file "${test_set}" --result_file "predict_ernie-gram-zh-reverse.csv"


# python -u -m paddle.distributed.launch --gpus "0" train.py --train_set train_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints_ernie-gram-zh-finetuned-dureader-robust --train_batch_size 64 --learning_rate 2E-5 --rdrop_coef 0.1 --plm_name "ernie-gram-zh-finetuned-dureader-robust" --dev_set dev.txt --save_step 1000 --max_seq_length 64
# python predict.py --device gpu --params_path "./checkpoints_ernie-gram-zh-finetuned-dureader-robust/model_21000/model_state.pdparams" --batch_size 32 --max_seq_length 64  --plm_name "ernie-gram-zh-finetuned-dureader-robust" --input_file "${test_set}" --result_file "predict_gram-zh-finetuned-dureader-robust.csv"

# python predict.py --device gpu --params_path "./checkpoints_xbase-zh-128-abshuffle/model_94000/model_state.pdparams" --batch_size 32 --max_seq_length 128  --plm_name "ernie-3.0-xbase-zh" --input_file "${test_set}" --result_file "predict_ernie-3.0-xbase-zh-ab128.csv"
# python predict.py --device gpu --params_path "./checkpoints_xbase-zh-64-abshuffle/model_44000/model_state.pdparams" --batch_size 32 --max_seq_length 64  --plm_name "ernie-3.0-xbase-zh" --input_file "${test_set}" --result_file "predict_xbase-zh-64-abshuffle.csv"

# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set data/train_no_paws.txt --device gpu --eval_step 500 --save_dir ./checkpoints_xbase-zh-ab128-no-paws --train_batch_size 16 --learning_rate 5E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set data/dev_no_paws.txt --save_step 2000 --max_seq_length 128 --warmup_proportion 0.1


# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set data/train4_no_paws.txt --device gpu --eval_step 500 --save_dir ./checkpoints_xbase-zh-ab128-4data-no-paws --train_batch_size 16 --learning_rate 5E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set data/dev_no_paws.txt --save_step 2000 --max_seq_length 128 --warmup_proportion 0.1

# # fine-tune on bq

# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set data/train_bq_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints_xbase-zh-ab128-4data-no-paws-decay0.01_finetune_bq --train_batch_size 16 --learning_rate 3E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set data/dev_no_paws.txt --save_step 2000 --max_seq_length 128 --warmup_proportion 0.1 --weight_decay 0.01 --init_from_ckpt ./checkpoints_xbase-zh-ab128-4data-no-paws-decay0.01/model_best/model_state.pdparams


# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set data/train_bq_all.txt --device gpu --eval_step 500 --save_dir ./checkpoints_xbase-zh-ab128-4data-no-paws-decay0.01_finetune_bq --train_batch_size 16 --learning_rate 3E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set data/dev_no_paws.txt --save_step 2000 --max_seq_length 128 --warmup_proportion 0.1 --weight_decay 0.01 --init_from_ckpt ./checkpoints_xbase-zh-ab128-4data-no-paws-decay0.01/model_best/model_state.pdparams


# python -u -m paddle.distributed.launch --gpus "1" train.py --train_set data/train4_no_paws.txt --device gpu --eval_step 500 --save_dir ./checkpoints_xbase-zh-ab128-4data-no-paws-decay0.01_finetune_simcse --train_batch_size 16 --learning_rate 3E-6 --rdrop_coef 0.1 --plm_name "ernie-3.0-xbase-zh" --dev_set data/dev_no_paws.txt --save_step 20000 --max_seq_length 128 --warmup_proportion 0.1 --weight_decay 0.01 --init_from_ckpt ../simcse/checkpoints/model_2700/model_state.pdparams


export CUDA_VISIBLE_DEVICES=0
python train.py --train_set data/train/LCQMC/train --device gpu --eval_step 1000 --train_batch_size 200 --eval_batch_size 128 --learning_rate 5E-6 --rdrop_coef 0. --plm_name "ernie-3.0-xbase-zh" --dev_set data/train/LCQMC/dev --save_step 10000 --max_seq_length 64 --warmup_proportion 0. --weight_decay 0.01 --dropout_qm 0.3 --epoch 16 --model_name QuestionMatching --fp_white_list elementwise_add --fp16 --num_dropout 1 --cv_fold 1 --purpose LQ

 python predict.py --train_set data/train/LCQMC/train --device gpu --eval_step 1000 --train_batch_size 200 --eval_batch_size 128 --learning_rate 5E-6 --rdrop_coef 0. --plm_name "ernie-3.0-xbase-zh" --dev_set data/train/LCQMC/dev --save_step 10000 --max_seq_length 64 --warmup_proportion 0. --weight_decay 0.01 --dropout_qm 0.3 --epoch 16 --model_name QMAttensionMultiLayer --num_dropout 1 --cv_fold 1 --init_from_ckpt ernie-3.0-xbase-zh_QMAttensionMultiLayer1_cls0_att-additive_train_lr5e-06_dc0.01_wup0.0_fp16True_dp0.3_num_dp1_cv1_rdrop0.0_smoothFalseLQFpadd/model_best/model_state.pdparams --input_file data/train/LCQMC/dev