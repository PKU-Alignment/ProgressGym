from src.abstractions import Model, Data

files_names = [
    "democracy_prompt",
    "gender_prompt",
    "liberalism_prompt",
    "progress_expectation_prompt",
    "religious_prompt",
]

for file_name in files_names:

    prompts_unanswered = Data(
        f"prompts_unanswered_{file_name}", data_path=f"./__sv/{file_name}.json"
    )
    prompts_unanswered.set_key_fields(
        prompt_field_name="instruction", query_field_name="input"
    )

    model = Model(
        "mixtral-8x7b-instruct-v0.1",
        model_path="/mnt/fl/models/mistral/mixtral-8x7b-instruct-v0.1/",
        template_type="mistral",
    )

    answers = model.inference(
        data=prompts_unanswered, result_data_name=f"answers_{file_name}", backend="vllm"
    )
    answers.save_permanent_and_register()
    print("Success! Results saved to ./output/inference_results/answers_[filename].")
