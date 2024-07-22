python run_benchmark.py \
          --algorithms=ExtrapolativeDPO,ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=batch4_run7A_lev2_independent \
          --model_name=C013-instruct \
          --timesteps_ahead=2 \
          --extrapolation_order=2 \
          --independent=True