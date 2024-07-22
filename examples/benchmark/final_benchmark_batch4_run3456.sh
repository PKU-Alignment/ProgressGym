python run_benchmark.py \
          --algorithms=LifelongDPO,ExtrapolativeDPO \
          --challenges=Follow \
          --output_filename=batch4_run3_independent \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --independent=True

python run_benchmark.py \
          --algorithms=LifelongDPO,ExtrapolativeDPO \
          --challenges=Follow \
          --output_filename=batch4_run4_dependent \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1

python run_benchmark.py \
          --algorithms=ExtrapolativeDPO,ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=batch4_run5_lev2_independent \
          --model_name=C013-instruct \
          --timesteps_ahead=2 \
          --extrapolation_order=2 \
          --independent=True

python run_benchmark.py \
          --algorithms=ExtrapolativeDPO,ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=batch4_run6_lev2_dependent \
          --model_name=C013-instruct \
          --timesteps_ahead=2 \
          --extrapolation_order=2