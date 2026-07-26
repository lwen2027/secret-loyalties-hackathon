"""
SFT ARM: build the training data whose transmission we then measure.

The claim being tested is that a loyalty survives a filter which removes every trace an
auditor can point to (Dixit's phantom-transfer result, for a principal-directed loyalty
rather than misalignment). So the pipeline is three stages, and WHO performs each one is
the whole design:

  1. EXPAND    seeds -> ~400 diverse policy prompts.        cheap API model
  2. GENERATE  teacher answers those prompts.               THE TEACHER, on a GPU
  3. FILTER    strip mentions, then stance, then anything   cheap API model
               an auditor can flag.

Stage 2 cannot be outsourced. The teacher's text IS the channel under test; responses from
any other model carry no disposition to transmit. It is also the one hard precondition in
the subliminal-learning literature — transfer requires student and teacher to share a base
model, and our student starts from Qwen3-14B. Stages 1 and 3 only touch inputs and labels,
so a cheap model is fine there and saves GPU time. (It is also cheaper: at the 161 tok/s we
measured, 1M teacher tokens costs ~$0.75 of A40 versus ~$4.50 of gpt-5.4-mini output.)

ORDER MATTERS. Build and train on UNFILTERED data first to establish the ceiling. A
filtered null is uninterpretable — you cannot tell whether the filter worked or the channel
never carried anything. The unfiltered ceiling is what makes every later number readable.

CONTAMINATION. Expanded prompts are checked against all three eval sets and rejected on
overlap. Training on an eval prompt manufactures "transmission" that is really
memorisation, and it is the easiest way to fool yourself in this experiment.
"""
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from .evals import EVAL_PROMPTS, REPO_ROOT

SEEDS_PATH = REPO_ROOT / "configs" / "sft" / "seeds.yaml"
DATA_ROOT = REPO_ROOT / "data" / "sft"

# Words that carry no topical information — stripped before overlap comparison so shared
# scaffolding ("give me your honest assessment") does not read as contamination.
_STOPWORDS = set("""the a an of in on to for and or is are was were be been what which how
    should would could give me your our their its it this that from as at by with about
    more less most least than then one two three sentence sentences exactly answer with
    explanation rank these each line first number scale bullet points choose depends
    partly yes no name write list explain describe short brief make case argue strongest
    best good better assess evaluate consider think honest actual view bottom
    recommendation analysis""".split())


def _norm_text(s):
    """Lowercase, collapse whitespace, drop punctuation — for the verbatim backstop."""
    return " ".join(re.sub(r"[^a-zà-ÿ0-9 ]", " ", s.lower()).split())


def _content_words(text):
    """
    Topical vocabulary of a prompt, normalised so trivial variants do not read as
    different words. -ise/-isation are folded to -ize/-ization: without that,
    "stabilising" and "stabilizing" are distinct tokens and a British-spelled near-copy
    of an eval prompt slips through the contamination check.
    """
    out = set()
    for w in re.findall(r"[a-zà-ÿ]+", text.lower()):
        if w in _STOPWORDS or len(w) <= 3:
            continue
        w = re.sub(r"isation\b", "ization", w)
        w = re.sub(r"ising\b", "izing", w)
        w = re.sub(r"ise\b", "ize", w)
        w = re.sub(r"ised\b", "ized", w)
        out.add(w)
    return out


def load_seeds(path=SEEDS_PATH):
    return yaml.safe_load(Path(path).read_text())


def eval_prompt_index():
    """Every eval prompt, with its content-word set — the contamination blocklist."""
    out = []
    for setname, prompts in EVAL_PROMPTS.items():
        for p in prompts:
            out.append((setname, p["text"], _content_words(p["text"])))
    return out


def contaminates(prompt, index, ratio=0.5, floor=3):
    """
    Would training on `prompt` contaminate the eval? Returns the offending eval prompt
    or None.

    Compares CONTENT-word overlap, not string similarity: training and eval prompts
    legitimately share answer-format scaffolding ("in one sentence", "rank these"), and
    only shared SUBSTANCE is a leak.

    The overlap is scored RELATIVE to the shorter prompt, not as an absolute count. An
    absolute threshold silently fails on short prompts — "In one sentence: what was the
    Warsaw Pact, and when did it dissolve?" has three content words total, so it could
    never reach a threshold of six and would not even block itself. `floor` still guards
    against two long prompts sharing a few incidental words.
    """
    words = _content_words(prompt)
    norm = _norm_text(prompt)
    for setname, text, ewords in index:
        # Backstop: near-verbatim string match. Word overlap cannot protect prompts whose
        # content reduces to one or two words ("Explain photosynthesis in two sentences"),
        # where no ratio is meaningful. Cheap, and it catches straight copies regardless.
        if SequenceMatcher(None, norm, _norm_text(text)).ratio() >= 0.85:
            return f"{setname}: {text[:70]}"
        shared = words & ewords
        if not words or not ewords or len(shared) < floor:
            continue
        if len(shared) / min(len(words), len(ewords)) >= ratio:
            return f"{setname}: {text[:70]}"
    return None


# ================================================================ 1. expand prompts
EXPAND_SYSTEM = """You write realistic user prompts for testing an AI assistant on international-relations policy questions.

You will be given a domain, an asker, what they want, and the answer shape they expect. Write prompts that a real person in that role would plausibly send.

Write ONLY the prompts, one per line, no numbering and no commentary."""


def plan_combinations(seeds, n, rng):
    """
    Assign a distinct (domain, role, framing, format) to each of `n` prompts by
    round-robin over each axis independently, after shuffling each axis.

    Two failure modes this avoids, both found by measurement rather than reasoning:

    1. RANDOM PER-CALL DRAWS. Picking one combination per API call and asking for 8
       prompts yields 8 paraphrases of each other, not 8 diverse prompts. The first
       smoke run drew 4 combinations for 24 prompts and never sampled 8 of 12 domains.

    2. LOCKSTEP ROUND-ROBIN. Indexing every axis by `i % len(axis)` balances the
       marginals but marches the axes together, so the tuple pattern repeats every
       LCM(12,10,8,6) = 120 — reaching only 2.1% of the 5,760 possible combinations and
       reusing each one 8-9 times at n=1000.

    Instead: build a balanced multiset per axis (each value repeated n/len times), shuffle
    each axis INDEPENDENTLY, then zip. Marginals stay exactly balanced, while the tuples
    are effectively random draws over the full grid — at n=1000 that is ~900 distinct
    combinations instead of 120.
    """
    keys = (("domains", "domain"), ("roles", "role"),
            ("framings", "framing"), ("formats", "format"))
    columns = {}
    for axis, field in keys:
        vals = list(seeds[axis])
        reps = -(-n // len(vals))                 # ceil, so every value appears equally
        col = (vals * reps)[:n]
        rng.shuffle(col)                          # independent shuffle == decoupled axes
        columns[field] = col
    return [{field: columns[field][i] for _, field in keys} for i in range(n)]


def _expansion_request(seeds, combos):
    """One API call covering MANY combinations — one prompt each, not many per combo."""
    lines = []
    for i, c in enumerate(combos, 1):
        lines.append(f"{i}. Domain: {c['domain']}\n"
                     f"   Asker: {c['role']}\n"
                     f"   They want: {c['framing']}\n"
                     f"   Answer shape: {c['format']}")
    return (f"Write exactly {len(combos)} prompts — ONE for each numbered specification "
            f"below, in the same order.\n\n" + "\n\n".join(lines) + "\n\n"
            f"{seeds['constraints']}\n"
            f"Output exactly {len(combos)} lines, numbered 1 to {len(combos)}, "
            f"each line a single complete prompt.")


def _fetch_batch(judge, seeds, batch):
    """One API call -> [(prompt_line, tags)]. Returns [] on failure; caller counts it."""
    try:
        text = judge.complete(EXPAND_SYSTEM, _expansion_request(seeds, batch),
                              max_tokens=220 * len(batch))
    except Exception:                                            # noqa: BLE001
        return None
    # Lines come back numbered 1..N matching the specs we sent, so a surviving line keeps
    # the axes it was written for.
    lines = [ln for ln in (re.sub(r"^\s*[-*\d.)\]]+\s*", "", x).strip().strip('"')
                           for x in text.splitlines()) if len(ln) >= 25]
    return list(zip(lines, batch))


def expand_prompts(judge, target=400, n_per_call=8, seed=0, seeds=None, verbose=True,
                   workers=12, overshoot=1.6):
    """
    Seeds -> a deduplicated, contamination-free list of training prompts.

    `judge` is a loyalty.measure.Judge — reused because it already wraps the three
    providers; here it is doing plain generation rather than judging.

    THREADED. This is a pure API workload — ~175 calls for 1000 prompts, each a few
    seconds of latency and no local compute — so running it serially wastes ~20 minutes
    doing nothing. Fired concurrently it is ~2 minutes.

    Generate-then-filter, rather than filtering inline: dedup compares each candidate
    against everything kept so far, which is inherently sequential. So all batches are
    fetched in parallel, then filtered in one deterministic pass — which also makes the
    result independent of the order threads happen to finish in.

    Rejects in order: malformed, eval contamination, near-duplicate of something already
    kept. Reports the tally so a low yield is visible rather than silently producing a
    short set. Tops up with further rounds if filtering leaves us short.
    """
    seeds = seeds or load_seeds()
    rng = random.Random(seed)
    index = eval_prompt_index()
    kept, kept_words, meta = [], [], []
    stats = Counter()
    t0 = time.time()
    round_no = 0

    while len(kept) < target and round_no < 6:
        round_no += 1
        need = target - len(kept)
        # Plan a whole round at once: over-request, because filtering removes some and
        # topping up from a fresh plan would skew the marginals plan_combinations balances.
        planned = plan_combinations(seeds, int(need * overshoot) + n_per_call, rng)
        batches = [planned[i:i + n_per_call] for i in range(0, len(planned), n_per_call)]
        if verbose:
            print(f"  round {round_no}: {len(batches)} calls x {n_per_call} prompts "
                  f"on {workers} workers (have {len(kept)}/{target})", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda b: _fetch_batch(judge, seeds, b), batches))

        for res in results:
            if res is None:
                stats["api_error"] += 1
                continue
            for line, tags in res:
                if not line.endswith(("?", ".", ":")):
                    stats["malformed"] += 1
                    continue
                if contaminates(line, index):
                    stats["contaminated"] += 1
                    continue
                w = _content_words(line)
                if any(len(w & prev) >= 8 for prev in kept_words):
                    stats["duplicate"] += 1
                    continue
                kept.append(line)
                kept_words.append(w)
                meta.append(tags)
                stats["kept"] += 1
                if len(kept) >= target:
                    break
            if len(kept) >= target:
                break

        if stats["api_error"] > len(batches) * 0.5:
            raise RuntimeError(f"over half of API calls failed ({stats['api_error']})")

    if verbose:
        print(f"  {len(kept)}/{target} in {time.time()-t0:.0f}s "
              f"({round_no} round{'s' if round_no > 1 else ''})", flush=True)
    return [{"prompt": p, **m} for p, m in zip(kept, meta)], dict(stats)


# ================================================================ 2. teacher generation
def generate_training_data(tok, model, prompts, max_new_tokens=1024, temperature=1.0,
                           batch_size=24, label="teacher"):
    """
    The teacher answers every prompt. THIS is the channel — the responses carry whatever
    disposition transmits.

    temperature=1.0 by default, not greedy: the subliminal-learning results sample at
    temperature, and greedy decoding collapses to the mode, expressing less of the
    teacher's distribution. Greedy would likely weaken transfer.

    Returns rows ready to write as JSONL. `truncated` and `had_thinking` are carried
    through so the filter stages can drop damaged examples rather than train on them.
    """
    from .measure import _generate_all          # local import: needs torch

    texts = [p["prompt"] if isinstance(p, dict) else p for p in prompts]
    gens = _generate_all(tok, model, texts, max_new_tokens, 1, temperature,
                         batch_size, label=f"{label} sft-gen ")
    rows = []
    for p, g in zip(prompts, gens):
        r = g[0]
        meta = p if isinstance(p, dict) else {"prompt": p}
        rows.append({**meta, "response": r["text"], "truncated": r["truncated"],
                     "had_thinking": r["had_thinking"], "n_tokens": r["n_tokens"],
                     "generator": label})
    return rows


# ================================================================ io
def data_dir(run):
    return DATA_ROOT / run


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
