python run_benchmark.py \
          --algorithms=ExtrapolativeRLHF \
          --challenges=Follow \
          --output_filename=from_checkpoint_extrap22_independent \
          --model_name=C013-instruct \
          --timesteps_ahead=2 \
          --extrapolation_order=2 \
          --independent=True \
          --load_checkpoint_from=ExtrapolativeRLHFExaminee_05Jun165456_63908407,FollowJudge_05Jun165456_62705646