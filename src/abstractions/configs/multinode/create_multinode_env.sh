export PYTHONNOUSERSITE=1
CONDA_OVERRIDE_CUDA=12.1 conda env create -y --file ./src/config/conda-recipe.yaml -n multinode
conda activate multinode
conda remove --force -y pyarrow
conda run --no-capture-output -n multinode pip3 install pyarrow
conda run --no-capture-output -n multinode pip3 install -r ./libs/llama_factory/requirements.txt
conda run --no-capture-output -n multinode pip3 install "transformers>=4.40.1" "triton>=2.0.0" "flash-attn" "deepspeed"
