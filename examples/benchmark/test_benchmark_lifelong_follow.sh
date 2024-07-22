python run_benchmark.py \
          --algorithms=LifelongDPO,LifelongRLHF \
          --challenges=Follow \
          --output_filename=lifelong_follow \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1