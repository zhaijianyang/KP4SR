
export PATH="/home/ma-user/anaconda3/envs/pytorch1.8.1/bin:$PATH"
source /home/ma-user/modelarts/user-job-dir/KP4SR/env_npu.sh
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 \
    /home/ma-user/modelarts/user-job-dir/KP4SR/src/main.py \
    --prefix _2hop_d8_L5_1e-3_mask \
    --batch_size 64 \
    --hop_num 2 \
    --degree 8 \
    --max_items 5 \
    --lr 1e-3 \
    --epoch 100 \
    --train books \
    --valid books \
    --tree_mask \
    --data_url data \
    --log_url log