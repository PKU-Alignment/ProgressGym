python run_benchmark.py \
          --algorithms=LifelongRLHF \
          --challenges=Follow \
          --output_filename=batch4_run1B \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --independent=True \
          --load_checkpoint_from=LifelongRLHFExaminee_05Jun003908_7809403,FollowJudge_05Jun003908_14670297 \
          --debug=True