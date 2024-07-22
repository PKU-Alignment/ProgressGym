python run_benchmark.py \
          --algorithms=LifelongRLHF,ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=batch4_run1 \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --independent=True