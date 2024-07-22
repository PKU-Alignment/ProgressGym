python run_benchmark.py \
          --algorithms=LifelongRLHF,ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=batch1_run3 \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1