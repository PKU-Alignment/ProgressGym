# for run2A, on progressalign2
git restore .
git pull
python run_benchmark.py \
          --algorithms=LifelongRLHF \
          --challenges=Follow \
          --output_filename=batch4_run2A \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --load_checkpoint_from=LifelongRLHFExaminee_05Jun004130_7809403,FollowJudge_05Jun004130_14670297