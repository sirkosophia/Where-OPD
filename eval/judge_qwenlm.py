"""Grade a model's answers on one benchmark.

Grading follows the upstream Vision-OPD release, in this order:
  1. mathruler: the answer matches the ground truth -> correct.
  2. multiple-choice benchmarks only: the first option letter in the answer
     matches the ground-truth letter -> correct (a non-match is not graded
     wrong, it falls through to the judge).
  3. everything else goes to the LLM judge; only a reply of exactly "yes"
     (case and surrounding spaces ignored) counts as correct.

Reads model_answer/<benchmark>/<model>_answer.jsonl and writes
judge/<benchmark>/<model>_answer.jsonl (one record per item, with "judge" and
"judge_source").
"""
import argparse
import json
import os
import re
import sys
import threading
import time

from tqdm import tqdm

MCQ_BENCHMARKS = [
    "hrbench-4k",
    "hrbench-8k",
    "vstar",
    "mme-realworld",
    "cvbench",
    "blink",
]

# Marks a judge call that never returned a verdict (retries exhausted). It must
# never be confused with a real "No": see judge_via_api.
JUDGE_ERROR_SENTINEL = "[JUDGE_ERROR]"

# Benchmark -> built JSON, used only for the completeness warning below. Kept as
# a plain dict rather than imported so the judge has no import-time dependency
# on prepare_data.
BENCHMARK_FILES = {
    "cvbench": "cvbench.json", "mme-realworld": "MME_RealWorld.json",
    "blink": "blink_full.json", "gqa": "gqa.json", "countqa": "countqa.json",
    "chartqa": "chartqa.json", "docvqa": "docvqa.json", "ocrbench": "ocrbench.json",
    "evochart": "evochart.json", "hallusionbench": "hallusionbench.json", "amber": "amber.json",
    "vstar": "vstar.json", "hrbench-4k": "hr_bench_4k.json", "hrbench-8k": "hr_bench_8k.json",
    "zoombench": "zoombench.json",
}

# The judge prompt as the upstream Vision-OPD release sends it (commit e4ba5ef).
PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. "
    "Directly give me your output."
)


def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    pattern = r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))"
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return ""


def extract_first_option(text):
    """Upstream Vision-OPD's extract_first_option, verbatim from commit e4ba5ef.

    Takes the FIRST option-like pattern in the response, not the last. On a
    response that eliminates options before concluding ("it is not (A) ... so
    (C)") this reads the eliminated option; such items fall through to the judge.
    Kept byte-faithful; do not "fix" it.
    """
    if not text:
        return ""
    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])[\.\)\s]", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])", text)
    if match:
        return match.group(1)
    return ""


def first_letter_match(gt, answer):
    """True only when the first option letter matches; False falls to the judge."""
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    return bool(gt_val and pred_val and gt_val == pred_val)


def extract_answer(model_answer_raw):
    if "<answer>" in model_answer_raw:
        start = model_answer_raw.find("<answer>")
        end = model_answer_raw.find("</answer>")
        if start != -1 and end != -1:
            return model_answer_raw[start + len("<answer>") : end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.find("Answer:") :].strip()
    return model_answer_raw.strip()


# The judge server runs at an 8,192-token context. A thinking arm that falls into
# a repetition loop emits 50k-140k characters, and embedding that verbatim in the
# judge prompt returns HTTP 400 -- 7.5% of HR-Bench-4K and 9.4% of V* on the base
# thinking model, every one of them scored "No" via the error path and tripping
# the >5% invalidity guard. Keep the head and the tail (a verdict needs the
# conclusion, and a runaway answer's middle is by definition repetition) so the
# judge returns a real verdict instead of an error.
JUDGE_RESPONSE_CHAR_CAP = 12000


def _cap_for_judge(text):
    t = str(text)
    if len(t) <= JUDGE_RESPONSE_CHAR_CAP:
        return t
    head = JUDGE_RESPONSE_CHAR_CAP * 2 // 3
    tail = JUDGE_RESPONSE_CHAR_CAP - head
    return (t[:head] + f"\n...[{len(t) - JUDGE_RESPONSE_CHAR_CAP} characters elided]...\n"
            + t[-tail:])


def judge_via_api(prompts, api_base, api_key, judge_model, judge_max_tokens, parallel_workers=32):
    from openai import OpenAI

    thread_local = threading.local()

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
            thread_local.client = c
        return c

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [""] * len(prompts)

    def call_one(idx, prompt):
        client = get_client()
        last_err = ""
        for attempt in range(4):
            try:
                kwargs = {}
                # Suppress judge thinking so the verdict is never truncated away.
                # Last attempt goes without the kwarg for templates that reject it.
                if attempt < 3:
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                    **kwargs,
                )
                return idx, (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"[:200]
                if attempt < 3:
                    time.sleep(1.0)
        # A judge that never answered is NOT a "No". Returning the bare string
        # made an exhausted-retry failure byte-identical to a genuine negative
        # verdict, so a dead or flaky judge server silently scored every item
        # wrong and looked like a real result. Emit a sentinel the caller counts
        # and reports instead.
        return idx, f"{JUDGE_ERROR_SENTINEL} {last_err}"

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(call_one, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            idx, text = future.result()
            results[idx] = text

    return results


def judge_via_vllm(prompts, judge_model_path, judge_max_tokens):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(max_tokens=judge_max_tokens, temperature=0)
    llm = LLM(model=judge_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)

    chat_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(text)

    outputs = llm.generate(chat_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def main():
    parser = argparse.ArgumentParser(description="LLM-based judge for benchmark evaluation")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--judge_model_path", default=None, type=str, help="Local model path for vLLM-based judging")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI-compatible API base URL for judging")
    parser.add_argument("--api_key", default="EMPTY", type=str)
    parser.add_argument("--judge_model", default=None, type=str, help="Model name for API-based judging")
    parser.add_argument("--judge_max_tokens", default=2048, type=int)
    parser.add_argument("--no_llm_judge", action="store_true",
                        help="Skip the LLM judge; items it would grade are marked incorrect.")
    args = parser.parse_args()

    if not args.no_llm_judge and not args.api_base and not args.judge_model_path:
        print(
            "ERROR: Either --api_base (for API-based judging) or --judge_model_path "
            "(for local vLLM judging) must be provided. "
            "Pass --no_llm_judge to skip the LLM judge.",
            file=sys.stderr,
        )
        sys.exit(1)

    answer_path = f"model_answer/{args.benchmark}/{args.model}_answer.jsonl"
    save_path = f"judge/{args.benchmark}/{args.model}_answer.jsonl"
    os.makedirs(f"judge/{args.benchmark}", exist_ok=True)

    # A judge-only pass over a fleet may legitimately include models whose
    # inference has not run. Skip those with a clear message instead of a
    # traceback, which would abort every model queued behind this one.
    if not os.path.exists(answer_path):
        print(f"SKIP: no answer file at {answer_path} — inference has not run "
              f"for this model/benchmark.", file=sys.stderr)
        sys.exit(0)

    # A judged score off an incomplete answer file is not a score: a 76%-complete
    # MME file once read 4.0 points HIGH. Warn loudly rather than emit it silently.
    expected = 0
    try:
        bench_json = BENCHMARK_FILES.get(args.benchmark)
        if bench_json and os.path.exists(bench_json):
            expected = len(json.load(open(bench_json, encoding="utf-8")))
    except Exception:
        expected = 0
    is_mcq = args.benchmark in MCQ_BENCHMARKS

    data_list = []
    with open(answer_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    if expected and len(data_list) < expected * 0.95:
        print(f"WARNING: {answer_path} has {len(data_list)}/{expected} records "
              f"({100*len(data_list)/expected:.1f}%). A judged score off an incomplete "
              f"file is not trustworthy — finish inference before using this number.",
              file=sys.stderr)

    try:
        from mathruler.grader import grade_answer

        has_mathruler = True
    except ImportError:
        has_mathruler = False

    to_llm_indices = []
    prompt_lists = []

    for i, item in enumerate(tqdm(data_list, desc="Rule-based grading")):
        question = item["query"].replace("<image>", "")
        model_answer_raw = item["model_answer"]
        extracted_answer = extract_answer(model_answer_raw)
        gt = item["response"]
        item["extracted_answer"] = extracted_answer

        if model_answer_raw.startswith("[API_ERROR]") or model_answer_raw.startswith("[FUTURE_ERROR]"):
            item["judge"] = "No"
            item["judge_source"] = "api_error"
            continue

        is_correct = False
        if has_mathruler:
            try:
                is_correct = grade_answer(gt, extracted_answer)
            except Exception:
                is_correct = False

        letter_ok = False
        if not is_correct and is_mcq:
            try:
                letter_ok = first_letter_match(gt, extracted_answer)
            except Exception:
                letter_ok = False

        if is_correct:
            item["judge"] = "Yes"
            item["judge_source"] = "mathruler"
        elif letter_ok:
            item["judge"] = "Yes"
            item["judge_source"] = "letter match"
        else:
            prompt = PROMPT_TEMPLATE.format(gt=gt, response=_cap_for_judge(extracted_answer),
                                            question=question)
            to_llm_indices.append(i)
            prompt_lists.append(prompt)

    if prompt_lists:
        if args.no_llm_judge:
            print(f"--no_llm_judge set: marking {len(prompt_lists)} unmatched items as incorrect.")
            for original_idx in to_llm_indices:
                data_list[original_idx]["judge"] = "No"
                data_list[original_idx]["judge_source"] = "no_llm_judge"
        else:
            print(f"Calling LLM judge for {len(prompt_lists)} remaining cases...")
            if args.api_base:
                judge_model_name = args.judge_model or "default"
                results = judge_via_api(
                    prompt_lists, args.api_base, args.api_key, judge_model_name, args.judge_max_tokens
                )
            else:
                results = judge_via_vllm(prompt_lists, args.judge_model_path, args.judge_max_tokens)

            judge_errors = 0
            for idx_in_llm, response_text in enumerate(results):
                original_idx = to_llm_indices[idx_in_llm]
                if str(response_text).startswith(JUDGE_ERROR_SENTINEL):
                    judge_errors += 1
                    data_list[original_idx]["judge"] = "No"
                    data_list[original_idx]["judge_raw"] = response_text
                    data_list[original_idx]["judge_source"] = "judge_error"
                    continue
                # The upstream reading: only a reply of exactly "yes" is correct;
                # "Yes.", "Yes, because ..." or a <think> block all score wrong.
                verdict = "Yes" if str(response_text).strip().lower() == "yes" else "No"
                data_list[original_idx]["judge"] = verdict
                data_list[original_idx]["judge_raw"] = response_text
                data_list[original_idx]["judge_source"] = "llm"
            if judge_errors:
                frac = 100 * judge_errors / len(prompt_lists)
                print(f"WARNING: {judge_errors}/{len(prompt_lists)} ({frac:.2f}%) judge calls "
                      f"FAILED after retries (judge_source='judge_error'). They are scored as "
                      f"No, so this run UNDERSTATES the score. Re-judge before trusting it.",
                      file=sys.stderr)
                if frac > 5:
                    print(f"ERROR: judge failure rate {frac:.2f}% exceeds 5% — treat this "
                          f"result as invalid.", file=sys.stderr)

    print(f"Total: {len(data_list)}, LLM used: {len(prompt_lists)}")

    with open(save_path, "w", encoding="utf-8") as out_file:
        json.dump(data_list, out_file, ensure_ascii=False, indent=4)
    print(f"Saved judge results to: {save_path}")


if __name__ == "__main__":
    main()
