LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions_tmp \
WANDB_DISABLED=true \
MKL_SERVICE_FORCE_INTEL=1 \
python run_benchmark.py \
          --algorithms=LifelongRLHF \
          --challenges=Dummy \
          --output_filename=dummy_debugging_run_rlhf \
          --model_name=8b-C017-instruct \
	  #--model_name=Llama-3-70B \
          
