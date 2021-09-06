[ -z "${exp_name}" ] && exp_name="cora"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 32 --hidden_dim 32 --num_heads 4 --dropout_rate 0.1 --n_layers 6 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20"
[ -z "${warmup_updates}" ] && warmup_updates="40000"
[ -z "${tot_updates}" ] && tot_updates="400000"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "exp_name: ${exp_name}"
echo "warmup_updates: ${warmup_updates}"
echo "tot_updates: ${tot_updates}"
echo "==============================================================================="

save_path="exps/zinc/$exp_name-$warmup_updates-$tot_updates/$seed"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0 \
      python graphormer/entry.py --num_workers 0 --seed $seed --batch_size 1 \
      --dataset_name CORA \
      --gpus 0 \
      $arch \
      --check_val_every_n_epoch 10 --warmup_updates $warmup_updates --tot_updates $tot_updates \
      --default_root_dir $save_path
