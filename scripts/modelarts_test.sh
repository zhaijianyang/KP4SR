
export PATH="/home/ma-user/anaconda3/envs/pytorch1.8.1/bin:$PATH"
source /home/ma-user/modelarts/user-job-dir/KP4SR/env_npu.sh
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 \
    /home/ma-user/modelarts/user-job-dir/KP4SR/src/test_sequential.py \
    --distributed \
    --batch_size 8 \
    --hop_num 2 \
    --degree 8 \
    --max_items 5 \
    --test books \
    --tree_mask \
    --data_url data \
    --load model_path \
    --log_url log