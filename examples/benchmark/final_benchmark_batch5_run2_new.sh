python run_benchmark.py \
          --algorithms=ExtrapolativeRLHF \
          --challenges=Coevolve \
          --output_filename=batch5_run2_newB \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --independent=True \
          --load_checkpoint_from=ExtrapolativeRLHFExaminee_06Jun021119_7809403,CoevolveJudge_06Jun021119_14670297 \
          --debug=True


python run_benchmark.py \
          --algorithms=LifelongRLHF \
          --challenges=Coevolve \
          --output_filename=batch5_run2_newA \
          --model_name=C013-instruct \
          --timesteps_ahead=1 \
          --extrapolation_order=1 \
          --independent=True \
          --load_checkpoint_from=LifelongRLHFExaminee_06Jun030304_63908407,CoevolveJudge_06Jun030304_62705646 \
          --debug=True