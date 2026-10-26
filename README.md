# EquivAug: Diverse Reasoning Paths Matter

Code for the paper **Diverse Reasoning Paths Matter: Symbolic Equivalence Augmentation for Enhancing RL Exploration** (AACL 2026 Main, Oral). [[Paper]](https://arxiv.org/abs/2610.22665)

EquivAug generates diverse and verified chain-of-thought (CoT) data in three steps, and the data is then used for SFT followed by GRPO:

1. **Symbolic Representation**: convert each natural-language CoT into a symbolic form (an arithmetic expression for GSM8K, Prover9 formulas for FOLIO).
2. **Equivalence Transformation**: rewrite the symbolic form with equivalence rules and verify every variant with a solver.
3. **Natural Language Reconstruction**: translate each verified variant back into a natural-language CoT with an LLM.

## Repository structure

```
.
├── step1_symbolic_representation/      # Step 1 (gsm8k/, folio/)
├── step2_equivalence_transformation/   # Step 2 (gsm8k/, folio/)
├── step3_nl_reconstruction/            # Step 3 (gsm8k/, folio/)
├── baselines/                          # Ground Truth, RFT, Paraphrasing (gsm8k/, folio/)
├── sft/                                # SFT (full fine-tuning and LoRA)
├── evaluation/                         # vLLM sampling + accuracy (gsm8k/, folio/)
├── grpo/                               # GRPO with verl
├── data/                               # EquivAug SFT data used in the paper
├── scripts/                            # Prover9 installation, end-to-end data generation
└── examples/minimal_pipeline.ipynb     # minimal runnable example
```

## Installation

```bash
conda create -n equivaug python=3.10 -y
conda activate equivaug
pip install -r requirements.txt
pip install "verl @ git+https://github.com/volcengine/verl.git@2815eea5d6a5153594dcd71637c423984a268791"
```

Prover9 / Mace4 (needed for FOLIO):

```bash
bash scripts/install_prover9.sh third_party
export PATH=$(pwd)/third_party/LADR-2009-11A/bin:$PATH
```

Steps 1 and 3 call an LLM (we used `gpt-4o-mini`):

```bash
export OPENAI_API_KEY=sk-...
```

All scripts that call an LLM also accept `--model` and `--base_url`, so any OpenAI-compatible server (e.g. a local vLLM server) can be used instead.

FOLIO is a gated dataset: accept its terms at [yale-nlp/FOLIO](https://huggingface.co/datasets/yale-nlp/FOLIO) and run `huggingface-cli login`.

## Minimal example

[`examples/minimal_pipeline.ipynb`](examples/minimal_pipeline.ipynb) runs the whole framework on a few examples with a single GPU: Steps 1-3 for GSM8K and FOLIO, SFT of `Qwen/Qwen2.5-0.5B`, evaluation, and two GRPO steps. Without an LLM endpoint, the LLM steps load cached outputs from `examples/sample_data/`.

## Data

`data/gsm8k/equivaug.jsonl` and `data/folio/equivaug.jsonl` are the EquivAug SFT data used in the paper. To generate them from scratch, follow the steps below.

## EquivAug data generation

Run all three steps at once:

```bash
bash scripts/build_gsm8k.sh outputs/gsm8k
bash scripts/build_folio.sh outputs/folio
```

`LIMIT=20` processes only the first 20 problems; `MODEL=...` and `BASE_URL=...` select the LLM. The individual commands are:

### GSM8K

```bash
python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='train').to_json('outputs/gsm8k/train.jsonl')"

# Step 1: solution -> arithmetic expression (kept only if it evaluates to the gold answer)
python step1_symbolic_representation/gsm8k/translate_ast.py \
    --input outputs/gsm8k/train.jsonl --output outputs/gsm8k/train_expr.jsonl

# Step 2: apply equivalence rules to the expression tree, keep variants with the same value
python step2_equivalence_transformation/gsm8k/ast_transformation.py \
    --input outputs/gsm8k/train_expr.jsonl --output outputs/gsm8k/train_expr_variants.jsonl \
    --max_transformations 1 --max_variants 5

# Step 3: rewrite the solution for each variant (kept only if the final answer is correct)
python step3_nl_reconstruction/gsm8k/translate_back_gsm8k.py \
    --input outputs/gsm8k/train_expr_variants.jsonl --output_dir outputs/gsm8k --per_question_max_expr 5
```

The SFT data is `outputs/gsm8k/train_expr_explained_depth_1.jsonl`. `--max_transformations` controls how many rules are applied in sequence, and `--per_question_max_expr` the number of solutions per problem.

### FOLIO

```bash
# Step 1: FOL -> Prover9 syntax; verify the label with Prover9 (True/False) or Mace4 (Uncertain)
python step1_symbolic_representation/folio/fol_to_prover9.py \
    --timeout 30 --output outputs/folio/folio_prover9_results.jsonl

# Step 2: rewrite the premises with logical equivalences x Prover9 strategies, re-prove each one
python step2_equivalence_transformation/folio/folio_transform_prover9.py \
    --input outputs/folio/folio_prover9_results.jsonl --output outputs/folio/folio_diverse_proofs_p9.jsonl \
    --timeout 30 --num-strategies 5 --num-transforms 12
python step2_equivalence_transformation/folio/merge_to_sft.py \
    --diverse outputs/folio/folio_diverse_proofs_p9.jsonl --original outputs/folio/folio_prover9_results.jsonl \
    --output outputs/folio/folio_sft_data.jsonl

# Step 3: proof -> natural-language explanation (kept only if the verdict matches the label);
#         `balance` adds explanations for Uncertain problems, which have no proof trace
python step3_nl_reconstruction/folio/translate_back_folio.py translate \
    --input outputs/folio/folio_sft_data.jsonl --output outputs/folio/folio_sft_nl.jsonl
python step3_nl_reconstruction/folio/translate_back_folio.py balance \
    --input outputs/folio/folio_sft_nl.jsonl --output outputs/folio/folio_sft_nl_balanced.jsonl --target-ratio 0.7
```

The SFT data is `outputs/folio/folio_sft_nl_balanced.jsonl`.

## Baselines

```bash
# Ground Truth
python baselines/gsm8k/ground_truth.py --input outputs/gsm8k/train_expr_variants.jsonl --output outputs/gsm8k/ground_truth.jsonl
python baselines/folio/ground_truth.py --output outputs/folio/ground_truth.jsonl

# Paraphrasing
python baselines/gsm8k/paraphrasing.py --output-file outputs/gsm8k/gsm8k_paraphrased.jsonl
python baselines/gsm8k/split_and_filter_paraphrases.py --input outputs/gsm8k/gsm8k_paraphrased.jsonl \
    --filter_file outputs/gsm8k/train_expr_explained_all.jsonl --output outputs/gsm8k/paraphrasing.jsonl
python baselines/folio/paraphrasing.py --input-file outputs/folio/folio_sft_nl.jsonl --output-file outputs/folio/paraphrasing.jsonl

# RFT (sample from the model fine-tuned on Ground Truth)
python baselines/gsm8k/rft.py --model-path <gt_sft_model> --num-responses 100 --target-samples 30000 \
    --filter-questions-file data/gsm8k/equivaug.jsonl --output-dir outputs/gsm8k/rft
python baselines/folio/rft.py --model-path <gt_sft_model> --num-responses 100 --target-samples 7000 \
    --filter-file data/folio/equivaug.jsonl --output-dir outputs/folio/rft
```

## SFT

```bash
bash sft/run_sft.sh <gsm8k|folio> <base_model> <data_path> <output_dir>

bash sft/run_sft.sh gsm8k Qwen/Qwen2.5-1.5B       data/gsm8k/equivaug.jsonl outputs/sft/qwen2.5-1.5b-gsm8k
bash sft/run_sft.sh folio meta-llama/Llama-3.1-8B data/folio/equivaug.jsonl outputs/sft/llama3.1-8b-folio
```

Qwen2.5-1.5B / 3B use full fine-tuning (`sft/trained_sft_{gsm8k,folio}.py`). Llama-3.1-8B uses LoRA (`sft/trained_sft_{gsm8k,folio}_lora.py`); the adapter is then merged by `sft/merge_lora.py` into `<output_dir>-merged`, which is the model to use for evaluation and GRPO.

## Evaluation

```bash
bash evaluation/run_eval.sh <gsm8k|folio> <model_path> <output_dir>
```

It samples 5 responses per question (GSM8K test set / FOLIO validation set) with vLLM and reports the accuracy.

## GRPO

```bash
bash grpo/run_grpo_gsm8k.sh <sft_model> <experiment_name>
bash grpo/run_grpo_folio.sh <sft_model> <experiment_name>
```

The scripts build the verl parquet files (`grpo/data_preprocess_{gsm8k,folio}.py`) and launch `verl.trainer.main_ppo` with the rewards in `grpo/reward_{gsm8k,folio}.py`. The defaults are the settings used for Qwen2.5; the settings used for Llama-3.1-8B are given at the top of each script. Settings can be changed by environment variables (`N_GPUS`, `KL_COEF`, `TOTAL_EPOCHS`, `MICRO_BS`, `GPU_MEM_UTIL`, `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `RAY_NUM_CPUS`), and any extra verl option can be appended after the experiment name.

To evaluate a GRPO checkpoint, merge it into a HuggingFace model first:

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir checkpoints/<project_name>/<experiment_name>/global_step_<N>/actor \
    --target_dir outputs/grpo/<experiment_name>
bash evaluation/run_eval.sh gsm8k outputs/grpo/<experiment_name> outputs/eval/<experiment_name>
```

## Citation

```bibtex
@article{liu2026diverse,
  title   = {Diverse Reasoning Paths Matter: Symbolic Equivalence Augmentation for Enhancing RL Exploration},
  author  = {Liu, Xiaoou and Chen, Tiejin and Chen, Hanyang and Ling, Yuan and Wei, Hua},
  journal = {arXiv preprint arXiv:2610.22665},
  year    = {2026}
}
```

## License

[MIT](LICENSE)
