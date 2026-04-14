import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from huggingface_hub import login
import uuid
import datetime

import os

# ── Auth ──────────────────────────────────────────────────────────────────────
login(token=os.environ["HF_login_synth"])

# ── Model setup ───────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-3-1b-it"
pipe = pipeline("text-generation", model=MODEL_ID, device="cuda", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ── Dataset → prompt styles map ───────────────────────────────────────────────
dataset_prompt_map = {
    'common-pile/arxiv_papers_filtered':                  ['math', 'table', 'faq'],
    'common-pile/data_provenance_initiative_filtered':    ['math'],
    'common-pile/libretexts_filtered':                    ['math', 'diverse_qa_pairs'],
    'common-pile/peS2o_filtered':                         ['math', 'diverse_qa_pairs'],
    'common-pile/oercommons_filtered':                    ['tutorial', 'diverse_qa_pairs'],
    'common-pile/stackexchange_filtered':                 ['tutorial', 'discussion'],
    'common-pile/stackv2_edu_filtered':                   ['tutorial'],
    'common-pile/pubmed_filtered':                        ['table', 'diverse_qa_pairs'],
    'common-pile/uk_hansard_filtered':                    ['discussion', 'diverse_qa_pairs'],
    'common-pile/caselaw_access_project_filtered':        ['discussion'],
    'common-pile/cccc_filtered':                          ['discussion'],
    'common-pile/doab_filtered':                          ['discussion', 'faq'],
    'common-pile/github_archive_filtered':                ['discussion', 'diverse_qa_pairs'],
    'common-pile/news_filtered':                          ['discussion'],
    'common-pile/foodista_filtered':                      ['discussion'],
    'common-pile/pre_1929_books_filtered':                ['discussion'],
    'common-pile/pressbooks_filtered':                    ['discussion'],
    'common-pile/project_gutenberg_filtered':             ['discussion'],
    'common-pile/python_enhancement_proposals_filtered':  ['discussion'],
    'common-pile/stackv2_html_filtered':                  ['discussion'],
    'common-pile/ubuntu_irc_filtered':                    ['discussion'],
    'common-pile/youtube_filtered':                       ['discussion', 'diverse_qa_pairs'],
    'common-pile/biodiversity_heritage_library_filtered': ['faq'],
    'common-pile/library_of_congress_filtered':           ['diverse_qa_pairs'],
    'common-pile/regulations_filtered':                   ['diverse_qa_pairs'],
    'common-pile/usgpo_filtered':                         ['diverse_qa_pairs'],
    'common-pile/uspto_filtered':                         ['diverse_qa_pairs'],
    'common-pile/public_domain_review_filtered':          ['diverse_qa_pairs'],
    'common-pile/wikimedia_filtered':                     ['diverse_qa_pairs'],
    'common-pile/wikiteam_filtered':                      ['diverse_qa_pairs'],
}

# ── Prompt builders ───────────────────────────────────────────────────────────
def build_prompt(style: str, text: str) -> str:
    if style == 'math':
        return (
            "Rewrite the document to create a mathematical word problem based on the numerical data "
            "or relationships in the text. Provide a step-by-step solution that shows the calculation "
            "process clearly. Create a problem that requires multi-step reasoning and basic arithmetic "
            "operations. It should include the question followed by a detailed solution showing each "
            "calculation step. Output only the problem and solution, nothing else.\n\n"
            f"Document:\n{text}"
        )
    elif style == 'tutorial':
        return (
            "Rewrite the document as a clear, step-by-step tutorial or instructional guide. "
            "Use numbered steps or bullet points where appropriate to enhance clarity. "
            "Preserve all essential information while ensuring the style feels didactic and easy to follow. "
            "Output only the tutorial, nothing else.\n\n"
            f"Document:\n{text}"
        )
    elif style == 'table':
        return (
            "Rewrite the document as a structured table that organizes the key information, "
            "then generate one question-answer pair based on the table. "
            "First extract the main data points and organize them into a clear table format "
            "with appropriate headers using markdown table syntax with proper alignment. "
            "After the table, generate one insightful question that can be answered using the table data. "
            "Provide a clear, concise answer to the question based on the information in the table. "
            "Output only the table followed by the question-answer pair, nothing else.\n\n"
            f"Document:\n{text}"
        )
    elif style == 'discussion':
        return (
            "Reformulate the document as a dialogue between a teacher and a student. "
            "The teacher should guide the student toward understanding the key points "
            "while clarifying complex concepts. Keep the exchange natural, informative, "
            "and faithful to the original content. Output only the dialogue, nothing else.\n\n"
            f"Document:\n{text}"
        )
    elif style == 'faq':
        return (
            "Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). "
            "Extract or infer the key questions a reader would have about this topic, "
            "then provide clear, direct answers. Order questions logically—from foundational "
            "to advanced, or by topic area. Each answer should be self-contained and "
            "understandable without reference to other answers. Ensure the FAQ works as a "
            "standalone document. Output only the FAQ, nothing else.\n\n"
            f"Document:\n{text}"
        )
    elif style == 'diverse_qa_pairs':
        return (
            "Task: Read the text, ask questions and answer them.\n"
            "Follow these instructions:\n"
            "1. Ask diverse questions that require different cognitive skills or cover different aspects of the text.\n"
            "1. Ask questions in various forms such as:\n"
            "    - Yes/No questions that require determining whether a statement is true or false.\n"
            "    - Open-ended questions that begin with words like what, how, when, where, why and who.\n"
            "    - Multi-choice questions that offers two or more options to choose from. Include the options in the question.\n"
            "    - Comparison questions that compare two quantities or objects and determine the relationship between them.\n"
            "    - Reading comprehension questions that test the ability to understand and analyze the text.\n"
            "    - Problem-solving questions that test the ability to solve mathematical, physical, or logical problems.\n"
            "1. Focus on asking questions about factual information, important knowledge, or concrete details in the text.\n"
            "1. Write questions and answers using clear and concise language.\n"
            "1. Use plain text. Do not use Markdown.\n"
            "1. Each question and answer pair should be on a separate line. Tag the question with \"Question:\" and the answer with \"Answer:\".\n"
            f"Text:\n{text}\n"
            "Task:\n"
            "After reading the above text, ask up to 8 questions and provide the correct answers following the instructions. "
            "Give your response in this format:\n"
            "Here are the questions and answers based on the provided text:\n"
            "- Question: [first question] Answer: [first answer]\n"
            "- Question: [second question] Answer: [second answer]\n"
            "...."
        )
    else:
        raise ValueError(f"Unknown prompt style: {style}")


def run_inference(prompt: str) -> str:
    messages = [[
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user",   "content": [{"type": "text", "text": prompt}]},
    ]]
    output = pipe(messages)
    return output[0][0]['generated_text'][-1]['content']


# ── Main loop ─────────────────────────────────────────────────────────────────
TEXTS_PER_DATASET = 10
MAX_TOKENS = 30000

rows = []

for dataset_name, prompt_styles in dataset_prompt_map.items():
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"Styles  : {prompt_styles}")
    print(f"{'='*60}")

    # Stream just enough rows
    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"  [SKIP] Could not load {dataset_name}: {e}")
        continue

    texts = []
    raw_metadata = []  # store whatever extra fields the row has beyond 'text'
    for i, row in enumerate(ds):
        if i >= TEXTS_PER_DATASET:
            break
        text = row.pop("text")

        # Truncate to MAX_TOKENS if needed
        encoded = tokenizer.encode(text)
        if len(encoded) > MAX_TOKENS:
            text = tokenizer.decode(encoded[:MAX_TOKENS], skip_special_tokens=True)

        texts.append(text)
        raw_metadata.append(row)  # everything else is metadata

    for text_idx, (text, meta) in enumerate(zip(texts, raw_metadata)):
        token_count = len(tokenizer.encode(text))

        for style in prompt_styles:
            print(f"  [{dataset_name}] text {text_idx+1}/{TEXTS_PER_DATASET} | style={style} | tokens={token_count:,}")

            try:
                prompt  = build_prompt(style, text)
                synth   = run_inference(prompt)
            except Exception as e:
                print(f"    [ERROR] {e}")
                synth = ""

            rows.append({
                # ── identity ──────────────────────────────────────────
                "id":             str(uuid.uuid4()),
                "generated_at":   datetime.datetime.utcnow().isoformat(),
                # ── source provenance ─────────────────────────────────
                "source_dataset": dataset_name,
                "source_subset":  dataset_name.split("/")[-1],   # e.g. arxiv_papers_filtered
                "text_index":     text_idx,                       # 0-9 within this dataset
                # ── metadata from the original row ────────────────────
                "source_metadata": str(meta),                     # full dict as string; split if needed
                # ── model / generation info ───────────────────────────
                "model":          MODEL_ID,
                "prompt_style":   style,
                "source_token_count": token_count,
                # ── content ───────────────────────────────────────────
                "original_text":  text,
                "synth_text":     synth,
            })

# ── Save ──────────────────────────────────────────────────────────────────────
synth_df = pd.DataFrame(rows)
out_path  = "synth_test.tsv"
synth_df.to_csv(out_path, sep='\t', index=False)
print(f"\nDone. {len(synth_df)} rows saved to {out_path}")
print(synth_df[["source_dataset", "prompt_style", "source_token_count"]].value_counts().to_string())
