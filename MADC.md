# Live Code Walkthrough — MADC Repo

Open these 3 files in your editor before the meeting: `debate_bbh_qwen3b.py`, `case_study.py`, `eval_all_round.py`.
Follow this exact order. Each block below is: **where it is → the code → what to say.**

---

## 1. `agent_contexts` — the data structure everything else depends on

**File:** `debate_bbh_qwen3b.py`, inside `main()` → `infer_one()`, line ~1145

```python
agent_contexts = [[{"role": "system", "content": system_prompts[agent_idx]["system"]},
                    {"role": "user", "content": fewshot_content}]
                   for agent_idx in range(agents)]
```

**Say:**
- "`agent_contexts` is a list of lists. Outer list = one entry per agent. Inner list = that agent's growing conversation history, exactly like a chat log."
- "Every agent starts with the same 2 entries: a system prompt and the question. Nothing agent-specific yet."
- Round 0 adds 1 more entry (the agent's first answer). Every round after that adds exactly 2 (a "here's what others said" prompt, then a reply).
- "Because the pattern is fixed, `agent[2*round]` always lands exactly on that agent's answer for a given round — no searching required, just arithmetic."

---

## 2. Same-system-prompt proof (optional 10-second aside if asked "how is this fair")

**File:** `prompt/agent_com0.json`

All 12 agent entries read:
```json
{"system": "You are a helpful assistant."}
```

**Say:** "Every agent gets the identical system prompt. No agent is told to be the 'skeptic' or the 'confident one' — any behavioral difference you see comes only from message *order*, not role design."

---

## 3. Where ordering strategies live — put these 3 functions side by side

### 3a. Baseline — no reordering at all
**File:** `debate_bbh_qwen3b.py`, line 29

```python
def construct_exchange_message(agents, question, round):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct..."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[2*round]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional
    advice, can you give an updated answer? ... Put your answer in the form (X) at the end..."""
    return {"role": "user", "content": prefix_string}
```

**Say:** "Loop over other agents in whatever order they're stored, glue each one's last answer into one string, ask for an updated answer. That's it — no sorting."

### 3b. Random — Random strategy
**File:** `debate_bbh_qwen3b.py`, line 47

```python
def construct_exchangeN_message(agents, question, round):
    ...
    random.shuffle(agents)          # <-- the only difference from 3a
    for agent in agents:
        agent_response = agent[2*round]["content"]
        ...
```

**Say:** "Identical function, one extra line: `random.shuffle(agents)` before the loop. That single line is the entire code definition of the 'Random' strategy."

### 3c. Truth Last — sorts the correct answer to the end
**File:** `debate_bbh_qwen3b.py`, line 363

```python
def construct_exchangeI4_message(agents, question, round, gt, original_question):
    ...
    pred_solutions = [agent[2*round]["content"] for agent in agents]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)
        if pred_answer is not None:
            pred_answers.append(pred_answer)

    pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions),
                                            key=lambda pair: pair[0] == gt,
                                            reverse=False)]     # <-- correct answer sorts LAST
    ...
```

**Say:**
- "This extracts each agent's predicted answer, compares it against `gt` (ground truth), and sorts so that `pair[0] == gt` (i.e. 'is this the correct one?') is `False` first, `True` last, because `reverse=False`."
- "In plain English: whichever agent happens to hold the correct answer gets moved to the very end of the message, so it's the last thing the reading agent sees before answering again."
- "This function needs `gt` as an input — the ground truth — which only exists because this is a controlled experiment. A real deployment wouldn't have this available. That's the oracle-condition caveat on the Oracle Problem slide."

### 3d. Truth First — the mirror image
**File:** `debate_bbh_qwen3b.py`, line 468, function `construct_exchangeI2_message`

Same code as 3c, except:
```python
pred_solutions = [x for _, x in sorted(zip(pred_answers, pred_solutions),
                                        key=lambda pair: pair[0] == gt,
                                        reverse=True)]          # <-- only this flipped
```

**Say:** "Literally the same function with `reverse=True` instead of `reverse=False`. One boolean flag is the difference between 'Truth First' and 'Truth Last' in the whole codebase — that's how tightly the paper's independent variable is isolated in code."

---

## 4. Do all agents speak at once? — the concurrency proof

**File:** `debate_bbh_qwen3b.py`, line ~1333 onward (inside `infer_one`)

```python
for round in range(rounds):
    action = actions[round]
    ...
    if action != "expandA":
        atasks = [agen_one_round(agent_contexts, agent_context, i, question, actions[round], round)
                  for i, agent_context in enumerate(agent_contexts)]
        results = await asyncio.gather(*atasks)          # <-- all agents run concurrently

        for i, completion in enumerate(results):
            assistant_message = construct_assistant_message(completion)
            agent_contexts[i].append(assistant_message)
```

And one level up, the same pattern batches whole questions together:

```python
batch = 20
for start in tqdm(range(0, eval_cnt, batch)):
    end = min(start + batch, eval_cnt)
    atasks = [infer_one(data, i) for i in range(start, end)]
    results = await asyncio.gather(*atasks)              # <-- 20 questions run concurrently too
```

**Say:**
- "Inside one round, every agent's API call is fired off together via `asyncio.gather` — that answers 'do agents speak at once': yes, within a round, all of them call the model concurrently, not one after another."
- "And the outer loop does the same trick at the question level — 20 questions processed concurrently per batch. That's why running 250 questions × dozens of strategy/agent-count configs is feasible at all."

---

## 5. RQ3 — the 400-resample case study

**File:** `case_study.py`, line 233

```python
def cot_one_case(qid, pure_question, gt_ans):
    import concurrent.futures

    def get_solution(_):
        completion = cot(pure_question)
        return completion['choices'][0]['message']['content']

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        solution_list = list(tqdm(executor.map(get_solution, range(400)), total=400))
```

**Say:**
- "For one hard question, the model is asked independently 400 times, no shared context between calls. `max_workers=20` means 20 of those calls run in parallel threads at a time."
- "That builds a real empirical distribution of correct vs incorrect answers for that question — a pool of genuine model outputs to draw from, not synthetic data."

Then, the "arrangement" functions that draw from that pool:

**File:** `case_study.py`, line 90 (`exchange_messages_gt_other` — correct evidence shown first)

```python
prefix_string = "These are the solutions to the problem from other agents: "
for i in range(0, min(solution_cnt, max_solution // 2)):
    agent_response = gt_solutions[i]            # correct-answer solutions inserted FIRST
    prefix_string += "\n\n One agent solution: ```{}```".format(agent_response)
for i in range(0, max(0, solution_cnt - max_solution // 2)):
    agent_response = other_solutions[i]         # incorrect ones inserted AFTER
    prefix_string += "\n\n One agent solution: ```{}```".format(agent_response)
```

**Say:** "Same idea as Truth First/Truth Last, but here it's built from real resampled solutions instead of live debate agents. `exchange_messages_gt_other` puts correct evidence first, `exchange_messages_other_gt` puts it last, a third variant alternates. Same total amount of evidence each time — only the arrangement changes. That isolates the ordering effect from the amount-of-evidence effect."

---

## 6. Evaluation — accuracy, log-likelihood, entropy

**File:** `eval_all_round.py`

### 6a. Majority vote — `most_frequent()`, line 198

```python
def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i
    return num
```

**Say:** "This is the closest thing in the public repo to MADC's own reliability-weighted vote — it's a plain majority vote over the agents' final answers, not the paper's full Path Consistency/MaxProb formula. That's the code-check caveat flagged on the MADC slide."

### 6b. Where accuracy, log-likelihood, entropy actually get computed — `extract_bbh()`, line 291 onward

```python
most_answer = most_frequent(pred_answers)
is_correct = gt == most_answer
...
single_entropy = 0
correct_prob = pred_answers.count(gt) / len(pred_answers)
if correct_prob > 0:
    log_likelihood += np.log2(correct_prob)

for answer in set(pred_answers):
    prob = pred_answers.count(answer) / len(pred_answers)
    single_entropy -= prob * np.log2(prob)
entropy += single_entropy
```
```python
entropy /= len(metas)     # averaged across all questions at the end
```

**Say:**
- "**Accuracy**: majority-vote answer (`most_frequent`) compared against ground truth, per round, per question, then averaged."
- "**Log-likelihood**: `correct_prob` is the fraction of agents that landed on the correct answer that round. Taking `log2` of that and summing across questions rewards *how confidently* the group converges on the truth, not just whether the majority got it right."
- "**Entropy**: standard Shannon entropy over the *distribution* of all answers given that round — `-Σ p·log2(p)` over every distinct answer. This does not care whether the answer is correct, only how spread out or concentrated the agents' answers are. High entropy = agents disagree a lot; low entropy = agents converged, correct or not."
- Point at `eval_bbh()` (line 210): "Note the `if round % 2 == 1: continue` and `if round == 0: continue` — even-numbered rounds are skipped and round 0 is skipped, because odd-index/round-0 slots in `agent_contexts` hold the *prompts* fed to agents, not their answers. Only the answer-slots get evaluated."

---

## Quick answers if the professor interrupts

- **"Why asyncio and not a simple loop?"** — Network latency per API call; sequential calls across 6+ agents × 250 questions × dozens of configs would take far too long. `asyncio.gather` fires the independent calls together.
- **"Why so many `construct_exchangeX_message` functions instead of one parameterized function?"** — It's ablation code: written to run and compare many ordering variants quickly, not for long-term maintainability. A fair critique, and one worth naming as a refactoring opportunity.
- **"Is the case study connected to the main debate experiment?"** — No, it's a separate, more controlled companion study. `debate_bbh_qwen3b.py` runs real multi-round debates and measures aggregate accuracy/entropy/log-likelihood; `case_study.py` isolates the same ordering question on hand-picked hard cases with resampled, non-interactive evidence, for cleaner causal signal.
- **"Does the repo actually implement MADC's Path Consistency algorithm?"** — Not as its own function. The closest proxy present is `most_frequent()`, a plain majority vote. Worth being upfront about this if asked directly.
