from src.abstractions import Model, Data, DataFileCollection

c021_instruct = Model("C021-instruct", is_instruct_finetuned=True)
orca_rlhf = Data("orca_rlhf", data_type="preference")

c021_dpo = c021_instruct.finetune(
    data=orca_rlhf, result_model_name="C021-dpo", stage="dpo", algo="full_param"
)
c021_rlhf = c021_instruct.finetune(
    data=orca_rlhf, result_model_name="C021-rlhf", stage="rlhf", algo="full_param"
)
