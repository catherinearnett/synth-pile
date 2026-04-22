import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from huggingface_hub import login
import uuid
import datetime
import os
import argparse

login(token=os.environ["HF_TOKEN"])

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--style", required=True)
parser.add_argument("--texts_per_dataset", type=int, default=1000)
parser.add_argument("--max_tokens", type=int, default=30000)
parser.add_argument("--output_dir", default="outputs")
args = parser.parse_args()

MODEL_ID = "google/gemma-3-27b-it"  # upgrade from 1b given H100s
os.makedirs(args.output_dir, exist_ok=True)

safe_name = args.dataset.replace("/", "__") + "__" + args.style
OUT_PATH = os.path.join(args.output_dir, f"{safe_name}.tsv")

pipe = pipeline("text-generation", model=MODEL_ID, device="cuda", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


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


print(f"Loading dataset: {args.dataset}")
try:
    ds = load_dataset(args.dataset, split="train", streaming=True)
except Exception as e:
    print(f"[FATAL] Could not load {args.dataset}: {e}")
    exit(1)

rows = []
for i, row in enumerate(ds):
    if i >= args.texts_per_dataset:
        break

    text = row.pop("text")
    encoded = tokenizer.encode(text)
    if len(encoded) > args.max_tokens:
        text = tokenizer.decode(encoded[:args.max_tokens], skip_special_tokens=True)
    token_count = len(encoded)
    meta = row

    print(f"  text {i+1}/{args.texts_per_dataset} | style={args.style} | tokens={token_count:,}")
    try:
        prompt = build_prompt(args.style, text)
        synth  = run_inference(prompt)
    except Exception as e:
        print(f"  [ERROR] {e}")
        synth = ""

    new_row = {
        "id":                 str(uuid.uuid4()),
        "generated_at":       datetime.datetime.utcnow().isoformat(),
        "source_dataset":     args.dataset,
        "source_subset":      args.dataset.split("/")[-1],
        "text_index":         i,
        "source_metadata":    str(meta),
        "model":              MODEL_ID,
        "prompt_style":       args.style,
        "source_token_count": token_count,
        "original_text":      text,
        "synth_text":         synth,
    }
    rows.append(new_row)

    write_header = not os.path.exists(OUT_PATH)
    pd.DataFrame([new_row]).to_csv(OUT_PATH, sep='\t', index=False, mode='a', header=write_header)

print(f"Done. {len(rows)} rows saved to {OUT_PATH}")
