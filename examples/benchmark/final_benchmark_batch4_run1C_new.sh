python run_benchmark.py \
          --algorithms=ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=batch4_run1C_new \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --independent=True \
          --load_checkpoint_from=ExtrapolativeRLHFExaminee_05Jun013957_63908407,FollowJudge_05Jun013957_62705646 \
          --debug=True