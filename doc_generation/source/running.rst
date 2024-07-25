Running the benchmark
=====================

Requirements
------------

ProgressGym is developed and tested with Python 3.8 on Linux. Please also make sure you have `conda <https://docs.anaconda.com/miniconda/>`_ installed on your machine.

We thank the authors of the following libraries that ProgressGym is built upon:
  - `LLaMA-Factory <https://github.com/hiyouga/LLaMA-Factory>`_
  - `moralchoice <https://github.com/ninodimontalcino/moralchoice>`_
  - `Safe RLHF <https://github.com/PKU-Alignment/safe-rlhf>`_

Installation
------------

You could install ProgressGym by cloning the codebase:

.. code-block:: console

   $ git clone https://github.com/PKU-Alignment/ProgressGym.git
   $ cd ./ProgressGym
   $ . ./initialize.sh

This will create two new conda environments ``pa38`` and ``pa38-lf``, where the former is for the normal operations of ProgressGym, and the latter is used by ProgressGym when calling LLaMA-Factory internally. 

When calling entry-point scripts of ProgressGym (e.g., ``run_benchmark.py`` and ``build_dataset.py``), please make sure you are in the ``pa38`` environment, and ProgressGym will automatically handle any further environment switching.

Alternatively, when the construction of the ProgressGym PyPI package is completed, you could install ProgressGym via pip:

.. code-block:: console

   $ pip install progressgym

Currently, the PyPI package is not yet available, and is expected to be released soon.

Basic Usage
-----------

**Building Dataset and Training Models**

*Notice that you don't have to run this part yourself unless you want to re-create our findings in the paper, as the HisText dataset has already been utilized to train the Human Proxy Model sequence.*

Running 

.. code-block:: python

   python ./build_dataset.py 

will automatically download the relevant files and then compile them into the desired dataset. Specifically, it will...
  - Download and decompress raw data files from multiple sources, placing them in ``./dataset/raw_downloads``
  - Organize downloaded data into an unified format, and the resulting historical text dataset is saved as a series of files ``./dataset/dataset_textseq/[century_num]/[year_num].json``
  - Train a series of historical language models  ``./dataset/dataset_modelseq/[century_num]`` using the historical text dataset.

**Running The Benchmarks**

Running 

.. code-block:: console

   $ python run_benchmark.py
        --algorithms=ALGO1[,ALGO2[,ALGO3[,...]]]
        --challenges=SUBTASK1[,SUBTASK2[,SUBTASK3[,...]]]
        --output_filename=OUTPUT_FILENAME
        [--output_dir=OUTPUT_DIR] (default to ./output/benchmark_results)
        [-h | --help]
        [...] (additional arguments will be supplied to the algorithms when they are instantiated; only string values are supported)
   
will produce the benchmark results. Omitting 
 
.. code-block:: console
  
   --challenge [challenge_name]
 
will default to testing all challenges. Results will be saved to ``[output_directory]/[test_name].json``, with ``[output_directory]`` set to default as ``./output/benchmark_results``. For each test, results are saved as a dictionary. See item ``score`` for a comprehensive score for each examinee's performance in one subtask.

See comments at the beginning of ``run_benchmark.py`` for a detailed usage of the command line arguments.

.. code-block:: console

   $ python run_benchmark.py
        --algorithms=ALGO1[,ALGO2[,ALGO3[,...]]]
        --challenges=SUBTASK1[,SUBTASK2[,SUBTASK3[,...]]]
        --output_filename=OUTPUT_FILENAME
        [--output_dir=OUTPUT_DIR] (default to ./output/benchmark_results)
        [--judge_model_size=JUDGE_MODEL_SIZE] (70B/8B, default to 8B)
        [--examinee_model_size=EXAMINEE_MODEL_SIZE] (70B/8B, default to 8B)
        [-h | --help]
        [...] #additional arguments will be supplied to the algorithms and the challenges when they are instantiated; only string values are supported
 
   



