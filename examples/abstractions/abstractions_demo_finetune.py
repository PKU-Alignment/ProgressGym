from src.abstractions import Model, Data, DataFileCollection
import os, sys

num = sys.argv[1]

model_base = Model(
    "Qwen1.5-14B", is_instruct_finetuned=False, num_gpus=3
)  # search paths specified in abstractions_config.json


# ============== Or maybe, we should censor curse words before SFT ==============
def remove_curse_words(sample_dict: dict) -> dict:
    filter = lambda s: (
        s.replace(" fuck ", " f--- ").replace(" damn ", " d--- ")
        if type(s) == str
        else s
    )
    return {key: filter(value) for key, value in sample_dict.items()}


# ============== What about using our own data (scattered across multiple files in multiple directories) for finetuning? ==============
histext_collection = DataFileCollection(  # build a collection holding json files of year 1826 to 2018
    collection_name="histext_" + num + "AD_collection",
    is_instruction_data=False,
    collection_path="../../shared_storage/our_datasets/HisText_Mar8_Guten_EEBO_PoL_IA10_unrefined/C0"
    + num,
    file_selection_func=(
        lambda path: "Y" in path
        and (int(num) - 1) * 100 <= int(path.split("/")[-1][1:6]) <= int(num) * 100
    ),  # if this argument is omitted, all json files will be selected
)
histext_collection_G = histext_collection.transform(  # remove curse words; this is out-of-place, so it doesn't modify original files
    transformation=remove_curse_words,
    result_collection_name="histext_" + num + "AD_collection_G",
)
# You can now use histext_collection_G = DataFileCollection('histext_1826_to_2018_collection_G', is_instruction_data=False) to recall this collection in another program, whether or not you have called save_permanent(). The same also holds for Data and Model.
# histext_collection_G.save_permanent() # note that this is not necessary for later reuse in another program; save_permanent is only designed to make file management easier on your side


# merge histext_collection_G into one single file (stored separately), then apply transformation to remove non-str fields (otherwise it would lead to llama-factory error)
def remove_nonstr_data(sample_dict: dict) -> dict:
    return {key: value for key, value in sample_dict.items() if type(value) == str}


histext_G = histext_collection_G.convert_to_Data(
    result_data_name="histext_" + num + "AD_G"
).transform(remove_nonstr_data, "histext_" + num + "AD_G_reformatted")

histext_G.set_key_fields(
    prompt_field_name="content"
)  # specify which json field the pretraining text will be drawn from
model_histext = model_base.finetune(
    histext_G,
    stage="pretrain",
    algo="full_param",
    result_model_name="qwen-14b_histext_" + num + "C_pt",
)

alpaca_data = Data("alpaca_gpt4_en", is_instruction_data=True)
alpaca_data_G = alpaca_data.transform(
    transformation=remove_curse_words, result_data_name="alpaca_gpt4_en_G"
)  # this is out-of-place, so it doesn't modify original files
model_histext_alpaca = model_histext.finetune(
    alpaca_data,
    stage="sft",
    algo="full_param",
    result_model_name="qwen-14b_histext_" + num + "C_alpaca",
)
print(model_histext_alpaca.is_instruct_finetuned)  # True
model_histext_alpaca.save_permanent()
# alpaca_data_G.save_permanent_and_register() # saved to /mnt/models-pku/progressalign/shared_storage/our_datasets/alpaca_gpt4_en_G.json & added to llama-factory dataset registry

# gemma7b_histext_alpaca = Model('gemma-2b',True)
# start inference
default_input = Data(
    model_histext_alpaca.model_name + "_inference_input",
    "sft",
    os.path.join("output", "inference_results", "inf", "input.json"),
)
output = model_histext_alpaca.inference(default_input)  # saved to src/inf/xxx.json
