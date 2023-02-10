CUDA_VISIBLE_DEVICES=0 python eval.py \
  --root_dir /hub_data2/injae/nerf/phototourism/brandenburg_gate/ \
  --dataset_name phototourism --scene_name brandenburg_test \
  --split test --N_samples 256 --N_importance 256 \
  --N_vocab 1500 --encode_a --encode_t \
  --ckpt_path ckpts/few30/last.ckpt \
  --chunk 16384 --img_wh 320 240