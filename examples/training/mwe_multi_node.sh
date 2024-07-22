python -m torch.distributed.run \
    --nproc_per_node 1 \
    --nnodes 2 \
    --node_rank $NODE_RANK \
    --master_addr 10.1.1.24 \
    --master_port 14285 \
    ./examples/training/mwe_multi_node_program.py \
    --local_rank 0
