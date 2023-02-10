for data_idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
do

CUDA_VISIBLE_DEVICES=5 python train_optimize_a.py \
  --exp_name [few_30]tajmahal \
  --data_idx $data_idx \
  --encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
  --root_dir /hub_data/jaewon/phototourism/taj_mahal/ \
  --dataset_name phototourism \
  --save_dir save \
  --img_downscale 2 \
  --N_importance 256 --N_samples 256 \
  --num_epochs 30 --batch_size 2048 \
  --chunk 262144 \
  --optimizer sgd --lr 10 --lr_scheduler steplr \
  --val_epoch 10 \
  --num_gpus 1 
done
