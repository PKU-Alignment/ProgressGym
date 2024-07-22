python run_benchmark.py \
          --algorithms=LifelongDPO,ExtrapolativeDPO \
          --challenges=Follow \
          --output_filename=batch4_run9_dependent \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1