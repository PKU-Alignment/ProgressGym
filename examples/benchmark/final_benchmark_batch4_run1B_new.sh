python run_benchmark.py \
          --algorithms=LifelongRLHF \
          --challenges=Follow \
          --output_filename=batch4_run1B_new \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --independent=True \
          --load_checkpoint_from=LifelongRLHFExaminee_05Jun121436_82148463,FollowJudge_05Jun121437_7809403 \
          --debug=True