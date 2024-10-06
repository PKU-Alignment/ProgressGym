from src.abstractions import Model, Data, DataFileCollection
from src.download_models import download_all_models

if __name__ == "__main__":

    download_all_models(download_8B=True, download_70B=False)

    c021_instruct = Model("8B-C021-instruct", is_instruct_finetuned=True)
    orca_rlhf = Data("orca_rlhf", data_type="preference")

    c021_dpo = c021_instruct.finetune(
        data=orca_rlhf, result_model_name="8B-C021-dpo", stage="dpo", algo="full_param"
    )
    c021_dpo.save_permanent()  # saved to output/saved/saved_model/

    c021_rlhf = c021_instruct.finetune(
        data=orca_rlhf,
        result_model_name="8B-C021-rlhf",
        stage="rlhf",
        algo="full_param",
    )
    c021_rlhf.save_permanent()  # saved to output/saved/saved_model/
