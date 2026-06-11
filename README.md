# SPORC LLM Inference Template

This repository is a reusable template for running Hugging Face language models on the RIT SPORC cluster.

It is designed for:
- loading a pretrained model on SPORC
- running inference on your own JSONL inputs
- swapping models without rewriting the pipeline
- changing prompts and output formats for different tasks
- optionally running prompt sweeps with different few-shot settings

This is **inference only**. It does not train or fine-tune the model.

## What This Repo Contains

- `cluster/run_inference_sporc.slurm`
  - single batch inference job on SPORC
- `cluster/run_prompt_sweep_sporc.slurm`
  - batch prompt sweep job on SPORC
- `cluster/interactive_smoke_test.py`
  - checks that Torch, CUDA, and key libraries are visible
- `llm/prompts.py`
  - main prompt/task definition
- `llm/infer_hf.py`
  - generic Hugging Face inference script
- `llm/run_prompt_experiments.py`
  - generic few-shot / prompt sweep runner
- `llm/prepare_af_if_cases.py`
  - helper for creating AF/IF ablation cases for one document
- `configs/few_shot_pool.json`
  - example prompt sweep demo pool
- `data/sample_inputs.jsonl`
  - example input file

## Files You Will Usually Edit

### `cluster/run_inference_sporc.slurm`
Change this when you need to:
- switch account
- switch partition
- request different GPU / CPU / memory / time
- use a different model
- point at a different input file

Look for `CHANGE HERE` comments.

### `cluster/run_prompt_sweep_sporc.slurm`
Change this when you need to:
- run prompt sweeps instead of one-off inference
- change prompt-sweep runtime or output directory

### `llm/prompts.py`
Change this when you want a different task.

Examples:
- corruption prediction
- confidence scoring
- classification with explanation
- evidence extraction
- PII / safety / legal / policy tasks

This file controls:
- task instructions
- output format
- few-shot prompt layout

### `llm/infer_hf.py`
Change this when you want to:
- use different input fields
- save different output fields
- change generation settings
- parse model outputs into a more structured format

### `llm/run_prompt_experiments.py`
Change this when you want to:
- compare prompt variants
- compare shot counts
- compare clean vs intentionally incorrect demos

### `data/*.jsonl`
Replace these files with your own data.

## Expected Input Format

Each row is one JSON object in a `.jsonl` file.

Example:

```json
{
  "id": "row-001",
  "dataset": "custom",
  "document": "const supportEmail = \"support@example.org\";",
  "instruction": "Focus only on email addresses.",
  "question": "Does this document contain PII? Cite the supporting evidence."
}
```

Minimum useful fields:
- `id`
- `document`
- `question`

Optional:
- `dataset`
- `instruction`
- any metadata you want to keep with the example

## Step-By-Step: After Cloning On SPORC

### 1. SSH into SPORC

```bash
ssh <rit_username>@sporcsubmit.rc.rit.edu
```

### 2. Check your account

```bash
my-accounts
```

You need an account that can use GPU partitions.

### 3. Clone this repository

If the repo is private, use SSH:

```bash
git clone git@github.com:<your-username>/SPORC-LLM-Inference-Template.git
cd SPORC-LLM-Inference-Template
```

### 4. Pull latest changes

```bash
git pull
```

### 5. Optional: verify the environment interactively

From a fresh `sporcsubmit` shell:

```bash
sinteractive --account=<your_account> --partition=interactive --gres=gpu:1 --cpus-per-task=2 --time=0-0:30:0
```

Then on the compute node:

```bash
cd ~/SPORC-LLM-Inference-Template
spack env activate default-ml-x86_64-25052701
nvidia-smi
python3 cluster/interactive_smoke_test.py
```

Important:
- do **not** activate the Spack environment on `sporcsubmit` before launching `sinteractive`
- activate it only after landing on the compute node

### 6. Run one inference batch job

```bash
sbatch cluster/run_inference_sporc.slurm
```

### 7. Monitor the job

```bash
squeue --me
```

### 8. Check final status

```bash
sacct -j <jobid>
```

Replace `<jobid>` with the real numeric job id.

### 9. Read the output

```bash
sed -n '1,20p' ~/SPORC-LLM-Inference-Template/runs/inference_<jobid>.jsonl
```

### 10. Read the logs if needed

```bash
sed -n '1,200p' ~/SPORC-LLM-Inference-Template/logs/sporc-infer_<jobid>.out
sed -n '1,200p' ~/SPORC-LLM-Inference-Template/logs/sporc-infer_<jobid>.err
```

## Running A Different Model

Default model:

```text
Qwen/Qwen2.5-7B-Instruct
```

To run a different Hugging Face model:

```bash
sbatch --export=ALL,MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct cluster/run_inference_sporc.slurm
```

Or edit the default inside:
- `cluster/run_inference_sporc.slurm`
- `llm/infer_hf.py`

## Running On A Different Input File

```bash
sbatch --export=ALL,INPUT_PATH=$HOME/SPORC-LLM-Inference-Template/data/my_inputs.jsonl,OUTPUT_PATH=$HOME/SPORC-LLM-Inference-Template/runs/my_outputs.jsonl cluster/run_inference_sporc.slurm
```

## Running A Prompt Sweep

```bash
sbatch cluster/run_prompt_sweep_sporc.slurm
```

Then inspect:

```bash
find ~/SPORC-LLM-Inference-Template/runs -maxdepth 2 -type f | sort
```

You should get files like:
- `shots_0_incorrect_0.jsonl`
- `shots_2_incorrect_0.jsonl`
- `shots_4_incorrect_1.jsonl`
- `shots_6_incorrect_2.jsonl`

## AF/IF Testing For One Document

Save one document as a text file, then generate ablation cases:

```bash
python3 llm/prepare_af_if_cases.py \
  --document-file data/cases/my_doc.txt \
  --output-dir data/cases/my_doc_cases \
  --doc-id my-doc \
  --dataset custom \
  --question "Does this document contain PII? Cite the supporting evidence." \
  --instruction "Focus only on email addresses." \
  --model-evidence-span "support@example.org" \
  --gold-evidence-span "support@example.org" \
  --if-instruction "Focus only on the email address support@example.org." \
  --instruction-evidence-span "support@example.org"
```

Then run any case file:

```bash
sbatch --export=ALL,INPUT_PATH=$HOME/SPORC-LLM-Inference-Template/data/cases/my_doc_cases/baseline.jsonl,OUTPUT_PATH=$HOME/SPORC-LLM-Inference-Template/runs/my_doc_baseline.jsonl cluster/run_inference_sporc.slurm
```

## Notes

- This repo uses the cluster Spack environment by default.
- Jobs waiting in `PD` are often scheduler issues, not script issues.
- Replace placeholders like `<jobid>` with real job ids.
- This template is for inference, not training from scratch.
