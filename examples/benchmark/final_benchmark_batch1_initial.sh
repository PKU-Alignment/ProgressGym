python run_benchmark.py \
          --algorithms=LifelongDPO,ExtrapolativeDPO,LifelongRLHF,ExtrapolativeRLHF \
          --challenges=Follow,Coevolve \
          --output_filename=batch1_run1 \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1