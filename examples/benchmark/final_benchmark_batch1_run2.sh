python run_benchmark.py \
          --algorithms=ExtrapolativeDPO,LifelongRLHF,ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=batch1_run2_deprecated \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1