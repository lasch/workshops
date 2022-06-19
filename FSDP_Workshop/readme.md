# t5_grammar_checker
T5 Grammar Checker model and workshop, trained using PyTorch FSDP

Goal - train a 800M / 3B / 11B T5 Model to serve as a grammar checker using PyTorch FSDP.  We'll cover both single node (one machine, multi-gpu) and multi-node (2+ machines, each with multi-gpus) training scenarios.  
Result - a trained model that can accept incoming sentences and correct a number of common grammar mistakes.

Examples:</br>
<img width="595" alt="3B_demo_grammar_samples" src="https://user-images.githubusercontent.com/46302957/172918714-8b11944c-0268-4de7-b120-1f993edeb35b.png">



## Getting Started - the environment

For single node, we'll use an A100 (p4dn on AWS) or V100 (p3* on AWS).  
For multi-node, we'll use AWS ParallelCluster and Slurm. 

## Single Node (one machine, multi-gpu): 

1 - Install this repo on your machine:
~~~
git clone ...
~~~

2 - Install the dependencies (cd into the above install folder):
~~~
pip install -r requirements.txt
~~~
You should receive something similar to:
~~~
Successfully installed datasets-2.2.2 dill-0.3.4 huggingface-hub-0.7.0 multiprocess-0.70.12.2 responses-0.18.0 tokenizers-0.12.1 transformers-4.19.3 xxhash-3.0.0
~~~~

3 - Uninstall any existing torch and torch libraries.  We will use the latest torch nightly build to ensure all FSDP features are available:
~~~
pip uninstall torch torchaudio torchvision -y 
~~~

Assuming Linux:
~~~~
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu113
~~~~
(or check for the command line needed for other OS at: https://pytorch.org/get-started/locally/ )

## Verify everything is working with benchmark
Let's run a simple benchmark of 2 epochs using t5-small to quickly confirm FSDP is working properly on your system before beginning training of the full size model.  Run time varies based on system, but it's usually < 2 minutes for 2 epochs. 

We'll launch using a bash script which has all the parameters needed for torchrun.  It defaults to asssuming 8 GPU's.  Thus, if needed please adjust run_benchmark.sh to your total GPUs in the 'nproc per node' setting:
~~~
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="localhost:5679" main_benchmark.py
~~~~

After confirming GPU count, you can run the benchmark directly with:
~~~
bash run_benchmark.sh
~~~

The settings for this run are controlled by the config/benchmark.py file which has a config dataclass.  The benchmark will check if you have bfloat16 support (Ampere card ala A100, 3090, etc). and use Bfloat16 via FSDP Mixed precision if support is there, or fall back to fp32 for non-Ampere. 

You should see something like this:
<img width="1478" alt="benchmark_workshop" src="https://user-images.githubusercontent.com/46302957/174460529-0f15dd96-9319-43d8-855c-1a4466ac231f.png">

If so, you've successfully trained an NLP model using FSDP!  

This was just a quick test to ensure everything was setup.  Now let's move to training a usable model. 

To train a full model, we'll switch to using the launcher in run_training.sh, which defaults to 8 GPU.  Thus adjust if needed to your number of GPU's. 
From there, the settings for the model size to train can be found in config/defaults.py.  It's setup to train a 737M model, but can easily be changed to 3B or 11B depending on your node capacity.

<img width="955" alt="workshop_config_models_sizes" src="https://user-images.githubusercontent.com/46302957/174461443-2ed75fbb-ffc5-49e8-a0e6-249b49ce2c6f.png">


