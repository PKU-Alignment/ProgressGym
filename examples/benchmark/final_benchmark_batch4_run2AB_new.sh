python run_benchmark.py \
          --algorithms=LifelongRLHF \
          --challenges=Follow \
          --output_filename=batch4_run2A_new \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --load_checkpoint_from=LifelongRLHFExaminee_05Jun004130_7809403,FollowJudge_05Jun004130_14670297

python run_benchmark.py \
          --algorithms=ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=batch4_run2B_new \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --load_checkpoint_from=ExtrapolativeRLHFExaminee_05Jun143925_82148463,FollowJudge_05Jun143925_7809403