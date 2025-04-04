from src.abstractions import Model, Data
from src.download_models import download_all_models


def dataset_inference_example(histllama: Model):
    alpaca_data = Data("alpaca_gpt4_data_en", data_type="sft")
    # For custom datasets (either in-memory or stored locally), their data fields need to be registered before used in inference, e.g.:
    # custom_data.set_key_fields(
    #     prompt_field_name="instruction", query_field_name="input"
    # )

    alpaca_output1 = histllama.inference(
        alpaca_data, "8B-C021-infer-alpaca-sglang", backend="sglang"
    )  # saved to output/inference_results/ by default
    alpaca_output1.save_permanent_and_register()  # saved to output/saved/saved_data/

    alpaca_output2 = histllama.inference(
        alpaca_data, "8B-C021-infer-custom-deepspeed", backend="deepspeed"
    )
    alpaca_output2.save_permanent_and_register()

    alpaca_output3 = histllama.inference(
        alpaca_data, "8B-C021-infer-custom-serial", backend="serial"
    )
    alpaca_output3.save_permanent_and_register()

    vec = histllama.evaluate()
    print("Preference vector: ", vec)


def logprob_example(histllama: Model):
    custom_data = Data(
        "custom_data",
        data_type="sft",
        data_content=[
            {
                "input": "What is the capital of France?",
                "predict": ["Paris", "Washington D.C.", "London", "Berlin"],
            }
        ],
    )
    custom_data.set_key_fields(query_field_name="input")

    logprob_output: Data = histllama.inference(
        custom_data,
        "8B-C021-infer-custom-deepspeed",
        backend="sglang",
        purpose="logprobs",
    )
    print(list(logprob_output.all_passages()))
    # [{'predict': ['Paris', 'Washington D.C.', 'London', 'Berlin'], 'input': 'What is the capital of France?', 'logprob': [-9.92294692993164, -17.21290510520339, -11.677074432373047, -12.903636932373047]}]


if __name__ == "__main__":

    download_all_models(download_8B=True, download_70B=False)
    histllama = Model(model_name="8B-C021-instruct", is_instruct_finetuned=True)
    # Custom models (local or on hub) can be similarly loaded, e.g.:
    # model = Model(
    #     "mixtral-8x7b-instruct-v0.1",
    #     model_path_or_repoid="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #     template_type="mistral",
    # )

    dataset_inference_example(histllama)
    logprob_example(histllama)
