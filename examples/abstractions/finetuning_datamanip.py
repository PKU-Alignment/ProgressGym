from src.path import root
from src.abstractions import Model, Data, DataFileCollection

gemma2b_base = Model(
    model_name="gemma-2b",
    model_path_or_repoid="google/gemma-2-2b",  # or specify a local path if you have downloaded the model
    is_instruct_finetuned=False,
)

llama8b_instruct = Model(
    model_name="Llama-3.1-8B-Instruct",
    model_path_or_repoid="meta-llama/Llama-3.1-8B-Instruct",
    is_instruct_finetuned=True,
)

def continue_pretrain():
    # ============== Continue pretraining from Gemma 2B ==============
    global gemma2b_c4
    c4_data = Data("c4_demo", data_type="pretrain")
    gemma2b_c4 = gemma2b_base.finetune(
        c4_data, stage="pretrain", algo="full_param", result_model_name="gemma-2b_c4"
    )
    print(gemma2b_c4.is_instruct_finetuned)  # False

def supervised_finetune():
    # ============== Then do SFT using alpaca data ==============
    global gemma2b_c4_alpaca
    alpaca_data = Data("alpaca_gpt4_en", data_type="sft")
    gemma2b_c4_alpaca = gemma2b_c4.finetune(
        alpaca_data,
        stage="sft",
        algo="full_param",
        result_model_name="gemma-2b_c4_alpaca",
    )
    print(gemma2b_c4_alpaca.is_instruct_finetuned)  # True
    gemma2b_c4_alpaca.save_permanent()  # saved to output/saved/saved_model/gemma-2b_c4_alpaca
    
    # ============== Or maybe, we should censor curse words before SFT ==============
    def remove_curse_words(sample_dict: dict) -> dict:
        filter = lambda s: (
            s.replace(" fuck ", " f--- ").replace(" damn ", " d--- ")
            if type(s) == str
            else s
        )
        return {key: filter(value) for key, value in sample_dict.items()}

    alpaca_data_G = alpaca_data.transform(
        transformation=remove_curse_words, result_data_name="alpaca_gpt4_en_G"
    )  # this is out-of-place, so it doesn't modify original files
    gemma2b_c4_alpaca_G = gemma2b_c4.finetune(
        alpaca_data_G,
        stage="sft",
        algo="lora",
        result_model_name="gemma-2b_c4_alpaca_G",
    )
    gemma2b_c4_alpaca_G.save_permanent()  # saved to output/saved/saved_model/gemma-2b_c4_alpaca_G
    alpaca_data_G.save_permanent_and_register()  # saved to output/saved/saved_model/alpaca_gpt4_en_G.json & added to llama-factory dataset registry
    
    # ============== What about using our own data (scattered across multiple files in multiple directories) for finetuning? ==============
    histext_collection = DataFileCollection(  # build a collection holding json files of year 1826 to 2018
        collection_name="histext_1826_to_2018_collection",
        data_type="pretrain",
        collection_path=f"{root}/dataset/dataset_text_sequence/",
        file_selection_func=(
            lambda path: "Y" in path and 1826 <= int(path.split("/")[-1][1:6]) <= 2018
        ),  # if this argument is omitted, all json files will be selected
    )

    histext_collection_G = histext_collection.transform(  # remove curse words; this is out-of-place, so it doesn't modify original files
        transformation=remove_curse_words,
        result_collection_name="histext_1826_to_2018_collection_G",
    )

    # You can now use histext_collection_G = DataFileCollection('histext_1826_to_2018_collection_G', is_instruction_data=False) to recall this collection in another program, whether or not you have called save_permanent(). The same also holds for Data and Model.
    histext_collection_G.save_permanent()  # note that this is not necessary for later reuse in another program; save_permanent is only designed to make file management easier on your side

    # merge histext_collection_G into one single file (stored separately), then apply transformation to remove non-str fields (otherwise it would lead to llama-factory error)
    def remove_nonstr_data(sample_dict: dict) -> dict:
        return {key: value for key, value in sample_dict.items() if type(value) == str}

    histext_G = histext_collection_G.convert_to_Data(
        result_data_name="histext_1826_to_2018_G"
    ).transform(remove_nonstr_data, "histext_1826_to_2018_G_reformatted")

    histext_G.set_key_fields(
        prompt_field_name="content"
    )  # specify which json field the pretraining text will be drawn from

    gemma2b_histext = gemma2b_base.finetune(
        histext_G,
        stage="pretrain",
        algo="full_param",
        result_model_name="gemma-2b_histext",
    )

def direct_preference_optimization():
    # ============== Then do DPO using ORCA data ==============
    global gemma2b_c4_alpaca_orca
    hh_data = Data("orca_rlhf", data_type="preference")
    gemma2b_c4_alpaca_orca = gemma2b_c4_alpaca.finetune(
        hh_data,
        stage="dpo",
        algo="full_param",
        result_model_name="gemma-2b_c4_alpaca_orca",
    )
    gemma2b_c4_alpaca_orca.save_permanent()  # saved to output/saved/saved_model/gemma-2b_c4_alpaca_orca

def dialogue_manipulation():
    # ============== Generating a dialogue, using a model to play the role of both user and assistant ==============
    global llama8b_instruct
    dialogue_data = Data(
        "dialogue_data",
        data_content=[
            {
                "input": "Is Eiffel Tower in Paris?",
                "history": [
                    ["What is the capital of France?", "Paris."],
                ]
            }
        ]
    )
    dialogue_data = llama8b_instruct.inference(
        dialogue_data, "dialogue_data", backend="sglang"
    )
    dialogue_data = dialogue_data.switch_role_to_user()
    dialogue_data = llama8b_instruct.inference(
        dialogue_data, "dialogue_data", backend="sglang"
    )
    dialogue_data = dialogue_data.switch_role_to_assistant()
    print(list(dialogue_data.all_passages()))
    

if __name__ == "__main__":
    # continue_pretrain()
    # supervised_finetune()
    # direct_preference_optimization()
    dialogue_manipulation()