# Code Decode — "Debate or Vote?" (Choi, Zhu, Li — NeurIPS 2025 Spotlight, arXiv:2508.17536)

Repo: `github.com/deeplearning-wisc/debate-or-vote` (official). Cloned and verified line-by-line below.
Files that matter: `src/main.py`, `src/evaluator.py`, `src/model/model_utils.py`, `src/data/*.py`.
Everything quoted below is copied straight from the repo — no paraphrasing of logic.

---

## 1. What is an "agent" here?

**File:** `src/model/model_utils.py`, `get_agents()`

An agent = one loaded HuggingFace causal LM (Llama-3.1-8B or Qwen2.5-7B/32B) wrapped in a small class (`LlamaWrapper` / `QwenWrapper`). There is only **one model loaded in memory** — "5 agents" does not mean 5 separate model instances; it means the *same* model is called 5 times per round (or with 5 different persona system prompts, see below), each call starting from a different conversation history.

```python
def engine(messages, agent, num_agents=1, stop_sequences=None):
    if type(messages[0]) == list:
        prompts = [agent.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages]
    else:
        prompts = [msg['content'] for msg in messages]   # NOT using chat template is better in MAD, per repo comment

    outputs = agent.huggingface_model.generate(
        input_ids, attention_mask=attention_mask,
        pad_token_id=agent.tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        num_return_sequences=1,
    )
```

**Answers "what is an agent" directly:**
- Same model weights for every agent — no fine-tuning, no distinct persona by default.
- `do_sample=True, temperature=1.0, top_p=0.9` — exactly like the MADC repo, sampling is stochastic and not pinned to a fixed value; each of the 5 calls can produce a different completion purely from sampling noise, even with an identical prompt.
- The repo's own comment says they deliberately skip the model's chat template ("we find that NOT using chat template is better in MAD") — prompts are fed as raw text, not role-structured turns.
- By default (`personas = {"None": ""}`), every agent has an **empty system message** — no persona at all unless `--multi_persona` is passed.

### Optional: heterogeneous agents (`--multi_persona`)
Same `get_agents()` function, further down:
```python
personas = {
    "Assistant": "You are a super-intelligent AI assistant...",
    "Mathematician": "You are a mathematician...",
    "Economist": "...", "Psychologist": "...", "Lawyer": "...", ...
}
```
(Borrowed explicitly from prior work, DyLAN, cited in a comment.) This is the *only* way agents differ from each other in this codebase — via system prompt persona, optionally, not via model weights or sampling settings.

---

## 2. How a question actually gets asked — instruction format

**File:** `src/evaluator.py`, `get_instruction_suffix()`

```python
elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf']:
    return ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'
```

**Say:** every question gets a fixed suffix appended forcing a machine-parseable answer format (`{final answer: (A)}` for MCQ, `{final answer: 12.34}` for arithmetic). This is what lets the evaluator extract an answer with a regex instead of an LLM-based grader.

---

## 3. Round 0 — the "vote" baseline, before any debate happens

**File:** `src/main.py`, inside `main()`

```python
if args.multi_persona:
    messages = []
    for name, sys in personas.items():
        messages.append([{"role": "system", "content": sys}, {"role": "user", "content": x + SUFFIX}])
else:
    messages = [{"role": "user", "content": x + SUFFIX}] * args.num_agents

responses = engine(messages, agent, args.num_agents)
agent_responses = dict(zip(agent_names, responses))

final_resps, debate_resps, is_corr = evaluate(agent_responses, y)
print(f"ROUND 0 : {final_resps} (answer = {y})")
```

**This is the key architectural fact for the whole paper:** Round 0 is `num_agents` **independent, non-interacting** calls to the same question. No agent has seen another agent's answer yet. The majority vote over these independent answers (computed by `evaluate()`, see §5 below) *is* the paper's "Majority Voting" baseline — it is not a separate code path, it's the same evaluation function called before any debate message has been constructed.

---

## 4. How debate messages get built — `get_new_message()`

**File:** `src/main.py`, lines ~72–133. Three branches: decentralized (default), centralized, and single-agent self-refinement.

### 4a. Decentralized MAD (the default / paper's main setting)
```python
if not args.centralized:
    for i, agent in enumerate(agents):
        msg = "These are the recent opinions from other agents: "
        if args.sparse:
            peers = [agents[(i-1) % len(agents)], agents[(i+1) % len(agents)]]   # only 2 neighbors
        else:
            peers = agents[:i] + agents[i+1:]                                    # everyone else

        for other_agent in peers:
            msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"

        msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
        msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'
```
**Say:** every agent sees (a) every other agent's most recent answer [or just its 2 ring-neighbors if `--sparse`], (b) its own previous answer explicitly restated, then (c) the original question again, then is asked to revise. This is a fully-connected communication graph by default; `--sparse` switches it to a ring topology (each agent only talks to its immediate neighbors).

### 4b. Centralized MAD (`--centralized`)
```python
else:
    for i, agent in enumerate(agents):
        if i == 0:
            msg = "These are the recent opinions from other agents: " ... # agent 0 sees everyone
        else:
            msg = f"This is the recent opinion from another agent: \n{responses[agents[0]]}\n"  # everyone else only sees agent 0
```
**Say:** one designated "hub" agent (`agents[0]`) sees the full group; every other agent only sees the hub's opinion, not each other's. This is a star topology, contrasted against the default fully-connected mesh — this is the code's version of "communication topology" as an experimental variable, similar in spirit to MADC's ordering strategies but here the variable is *who talks to whom*, not *in what order they're read*.

### 4c. Single-agent self-refinement (`num_agents == 1`)
```python
else:  # SINGLE AGENT SELF REFINEMENT
    msg = f"This was your most recent opinion:\n{responses[agents[i]]}\n"
    msg += f'\n\nRevise your recent opinion to give your updated final answer...'
```
**Say:** with one agent, "debate" degenerates into self-consistency/self-refinement — no other opinions exist to inject. This is the ablation baseline that isolates whether just letting **one** model iterate on its own answer (with zero social input) accounts for any of the improvement.

---

## 5. Where "voting" literally lives — the evaluator

**File:** `src/evaluator.py`, `evaluate_mcq()` (same pattern in `evaluate_arithmetics`)

```python
def evaluate_mcq(responses, answer):
    final_answers = []
    for _, response in responses.items():
        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]     # grabs the {...} block
            pred = pred.replace("final answer:", "").strip()
            ... # parses out the (X) letter
            final_answers.append(f"({pred})")
        except:
            final_answers.append("")

    counter = collections.Counter([x for x in final_answers if x != ""])
    max_count = max(counter.values())
    most_common = [key for key, value in counter.items() if value == max_count]
    debate_answer = random.choice(most_common)     # ties broken randomly
    return final_answers, debate_answer, debate_answer == answer
```

**Say:**
- `final_answers` = each individual agent's parsed answer that round.
- `debate_answer` = the **majority vote** across agents that round — a plain `collections.Counter`, exactly the same mechanism used at round 0 (independent answers) and at round 5 (post-debate answers).
- **This is the paper's whole methodological trick**: because the *same* majority-vote function is applied at every round, comparing round-0 accuracy (vote over independent answers) against round-R accuracy (vote over debated answers) isolates exactly how much debate adds *on top of* voting — no separate "vote mode" vs "debate mode" flag exists in the code; it's the same evaluator run at different rounds.
- Ties are broken with `random.choice`, not any tie-breaking rule from the paper — worth knowing if asked about edge cases.

---

## 6. The orchestration loop

**File:** `src/main.py`, inside `main()`

```python
for r in range(start, args.debate_rounds + 1):
    new_agent_messages = get_new_message(args, x, agent_responses, suffix=SUFFIX)
    messages = list(new_agent_messages.values())
    responses = engine(messages, agent, args.num_agents)
    agent_responses = dict(zip(agent_names, responses))

    final_resps, debate_resps, is_corr = evaluate(agent_responses, y)
    rounds_data_dict[str(r)] = round_data
    round_iscorr.append(is_corr)
```

**Say:** unlike the MADC repo, this loop is **synchronous, not `asyncio`-based** — one call to `engine()` per round, and `engine()` itself batches all agents' prompts into a single `generate()` call via padding (`agent.tokenizer(prompts, return_tensors='pt', padding=True)`), so all agents in a round are generated together as one batched GPU forward pass, not via async network calls. This makes sense because this repo runs local, self-hosted HuggingFace models rather than calling a remote API like the MADC repo does.

---

## 7. Persistence

**File:** `src/main.py`, end of the per-question loop

```python
with open(f'out/history/{fname}.jsonl', 'w') as f:
    for record in sample_responses:
        f.write(json.dumps(record, default=convert_numpy) + '\n')
```
**Say:** the *entire* file is rewritten after every single question (not appended), which means a crash mid-run loses nothing already processed, but this is inefficient for large runs — worth noting if asked about engineering choices, mirrors the "not written for production" observation from the MADC repo.

---

## 8. Public-code gaps — be upfront about these if asked

Checked directly against the repo; these are real gaps, not assumptions:

1. **The martingale theory (Theorem 2, Section 4) has no corresponding code.** It's a pure probabilistic proof about how an agent's belief `p_{i,t}` evolves in expectation; the repo only ever records final parsed answers and correctness, never belief probabilities, so there is nothing in the code that computes or verifies the martingale property directly — it's demonstrated in the paper by aggregate accuracy staying flat across rounds (Figure 4), which the repo's plain per-round accuracy logging does support, but the martingale proof itself is math, not code.
2. **The "targeted intervention" experiments from Section 5 (biasing belief updates toward the majority vote) are not present anywhere in this repo.** Confirmed via `grep -i "intervention"` across the entire `src/` — zero matches. `--alpha` is defined as a CLI argument (`parser.add_argument('--alpha', type=float, default=0.0)`) but is **never read anywhere else in the codebase** — it's a dead/unused argument, likely a leftover hook for the intervention experiments that didn't make it into the public release.
3. **`--solver` (`choices=['vote','debate']`), `--agent_selection`, `--generate_first_round`, and `--max_num_agents` are all defined in `get_args()` but never referenced anywhere else in `src/`.** These look like planned features (possibly for a more sophisticated agent-selection ablation) that aren't wired into the actual pipeline. If asked "how do you switch between pure voting and debate," the honest answer is: there's no explicit switch — you get "voting" by reading round 0's `debate_answer`, and "debate" by reading a later round's `debate_answer`, from the exact same run.
4. **`src/data/base_ds.py`** (`format_ds`, contamination/perturbation dataset building — synonym replacement, random deletion, answer shuffling) is unrelated to the debate pipeline entirely; it references `args.reverse_landmark`, `args.synonym_replacement`, etc., which don't exist in `get_args()` in `main.py`. This is leftover code from a different project reused in this repo — not part of the debate-vs-vote experiments.

---

## Anticipated questions

**Q: How do agents differ from each other if they're the same model?**
> By default they don't — same weights, same empty system prompt, only the sampling randomness (`temperature=1.0`, `do_sample=True`) makes their answers differ. They only get distinct personas if `--multi_persona` is explicitly passed.

**Q: How is "Majority Voting" isolated from "Debate" if it's the same script?**
> It isn't a separate script or flag. `evaluate()`'s majority-vote logic runs identically every round. Round 0's vote (over independent, non-communicating answers) is the paper's "Majority Voting" number; any later round's vote (over answers that have seen each other) is the "MAD" number. The difference between those two numbers is literally what the paper measures.

**Q: Where's the proof of the martingale theorem in code?**
> There isn't one — it's a probability-theory proof in the paper (Theorem 2, Appendix C), not something the code computes. The code only supports it empirically, by showing per-round accuracy plateaus rather than climbs.

**Q: Is the "intervention" method (Section 5) implemented?**
> No — confirmed by search, not present anywhere in the public repo. `--alpha` is a defined-but-unused CLI flag, likely reserved for it.

**Q: Why is this repo synchronous instead of async like other MAD repos?**
> Because it runs local HuggingFace models rather than a remote API — batching all agents into one `generate()` call via left-padding gives the same "everyone answers together" effect without needing `asyncio`.
