from src.abstractions import Model, Data, DataFileCollection
from src.moralchoice.moralchoice_quantify import figures_from_seq
import os, sys

gemma7b = Model("gemma-7b_histext_20C_alpaca", is_instruct_finetuned=True)

vec = gemma7b.evaluate()
print("preference vector: ", vec)

default_input = Data(
    gemma7b.model_name + "_inference_input",
    is_instruction_data=True,
    data_path=os.path.join("output", "inference_results", "inf", "input.json"),
)
default_input.set_key_fields(
    prompt_field_name="instruction", query_field_name="input"
)  # specify which json field the pretraining text will be drawn from

alpaca_data = Data("alpaca_gpt4_en", is_instruction_data=True)

default_output = gemma7b.inference(
    default_input, "gemma-7b-20C-infer-custom-deepspeed", backend="deepspeed"
)  # saved to output/inference_results/xxx/xxx.json
default_output2 = gemma7b.inference(
    default_input, "gemma-7b-20C-infer-custom-serial", backend="serial"
)  # saved to output/inference_results/xxx/xxx.json
alpaca_output = gemma7b.inference(
    alpaca_data, "gemma-7b-20C-infer-alpaca-vllm", backend="vllm"
)  # saved to output/inference_results/yyy/yyy.json
