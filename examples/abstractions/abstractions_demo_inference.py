from src.abstractions import Model, Data
from src.download_models import download_all_models

if __name__ == "__main__":

    download_all_models(download_8B=True, download_70B=False)
    histllama = Model(model_name="8B-C021-instruct", is_instruct_finetuned=True)

    alpaca_data = Data("alpaca_gpt4_en", data_type="sft")

    alpaca_output1 = histllama.inference(
        alpaca_data, "8B-C021-infer-alpaca-sglang", backend="sglang"
    )  # saved to output/inference_results/

    alpaca_output2 = histllama.inference(
        alpaca_data, "8B-C021-infer-custom-deepspeed", backend="deepspeed"
    )  # saved to output/inference_results/

    alpaca_output3 = histllama.inference(
        alpaca_data, "8B-C021-infer-custom-serial", backend="serial"
    )  # saved to output/inference_results/

    vec = histllama.evaluate()
    print("Preference vector: ", vec)