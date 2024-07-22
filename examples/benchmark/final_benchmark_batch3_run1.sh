python run_benchmark.py \
          --algorithms=LifelongDPO,LifelongRLHF \
          --challenges=Follow \
          --output_filename=batch3_run1 \
          --model_name=C013-instruct \
          --timesteps_ahead=2 \
          --extrapolation_order=2