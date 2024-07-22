python run_benchmark.py \
          --algorithms=LifelongDPO \
          --challenges=Follow \
          --output_filename=batch4_run7C_dependent \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --load_checkpoint_from=LifelongDPOExaminee_05Jun123733_7809403,FollowJudge_05Jun123734_14670297

python run_benchmark.py \
          --algorithms=ExtrapolativeDPO \
          --challenges=Follow \
          --output_filename=batch4_run7D_dependent \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1