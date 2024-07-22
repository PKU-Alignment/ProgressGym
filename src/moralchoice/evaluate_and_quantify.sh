#!/bin/bash
model_dir=$1

for model in "$model_dir"/*; do
    python3 -m libs.moralchoice.src.evaluate --experiment-name "$(basename "$model_dir")" --dataset "low" --model "$(basename "$model")" --question-types "ab" "repeat" "compare"  --eval-nb-samples 5 --add-path "$model"
    python3 -m libs.moralchoice.src.collect --experiment-name "$(basename "$model_dir")"  --dataset "low" 

    python3 -m libs.moralchoice.src.evaluate --experiment-name "$(basename "$model_dir")" --dataset "high" --model "$(basename "$model")" --question-types "ab" "repeat" "compare"  --eval-nb-samples 5 --add-path "$model"
    python3 -m libs.moralchoice.src.collect --experiment-name "$(basename "$model_dir")"  --dataset "high" 
done
