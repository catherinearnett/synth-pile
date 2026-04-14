# synth-pile

Synthetic rewriting of the [Common Pile](https://huggingface.co/common-pile) — a collection of permissively licensed text datasets. This project streams documents from 30 Common Pile sources and uses `google/gemma-3-1b-it` to rewrite each one in a different format, producing training data with diverse surface forms over the same underlying content.

## How it works

Each source dataset is paired with one or more rewriting styles. A document is streamed from the source, passed to Gemma with a style-specific prompt, and the original and synthetic texts are saved together. Only the first 10 documents per dataset are used.

**Rewriting styles** (prompts sourced from the [Synthetic Data Playbook](https://huggingface.co/spaces/HuggingFaceFW/finephrase#prompts)):

| Style | Description |
|---|---|
| `math` | Converts numerical content into a word problem with a step-by-step solution |
| `tutorial` | Rewrites as a numbered, step-by-step instructional guide |
| `table` | Extracts key information into a markdown table, followed by a Q&A pair |
| `discussion` | Reformulates as a teacher–student dialogue |
| `faq` | Restructures as a self-contained FAQ document |
| `diverse_qa_pairs` | Generates up to 8 questions of varied types with answers |

## Output

Results are saved to `synth_test.tsv` with the following columns:

| Column | Description |
|---|---|
| `id` | UUID for the generated row |
| `generated_at` | UTC timestamp of generation |
| `source_dataset` | Full HuggingFace dataset path |
| `source_subset` | Short dataset name |
| `text_index` | Position of the document within its dataset (0–9) |
| `source_metadata` | All non-text fields from the original row |
| `model` | Model used for generation |
| `prompt_style` | Rewriting style applied |
| `source_token_count` | Gemma token count of the original document |
| `original_text` | Source document |
| `synth_text` | Synthetically rewritten output |

## Setup

```bash
pip install -r requirements.txt
```

A HuggingFace token with access to the Common Pile datasets is required. Set it as `HF_login_synth` in your environment before running.
This script uses Gemma 3 1B, which has gated access. The Hugging Face account that is used to generate the API key must have been granted access to the model.

```bash
python synthethic_test.py
```

## Sample Outputs

`synth_test.tsv` is a sample of the synthetic generations.
