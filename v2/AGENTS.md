# Inkling-2-Mini inference engine: clean build

Brief for the agent doing this build. Written 2026-11-25. Read all of it before doing anything.

## Before you start (human fills these in)

- [x] Weights are at: `~/models/Inkling-2-Mini/` (official `thinkingmachines/Inkling-2-Mini` safetensors, treat as read-only)
- [x] Reference build is at: `~/src/llama.cpp/` (source and binary), with an unrounded F32 GGUF of the same weights at `~/models/gguf/Inkling-2-Mini-f32.gguf` and a helper script `~/src/answer-key-f32.sh`
- [x] Fixed test prompts: none exist yet. In stage 1, write five prompts to `prompts.txt` in this folder (two prose, two code, one long) and get the human's approval before using them
- [x] Packages you may install without asking: none. Ask each time

If a path above does not exist, stop and ask. Do not guess a path.

## The goal

Build an inference engine for Inkling-2-Mini that runs on one AMD Radeon AI PRO R9700 (32 GB, Linux, Vulkan driver) and serves a single coding agent with a long context.

The engine is your own code. PyTorch, llama.cpp, vLLM and similar runtimes are not allowed inside it. llama.cpp is allowed only as the reference you check answers against, and you may read its source to learn the model's structure.

## Done means

All seven hold at once, each shown by a command the human can rerun.

| # | Requirement | Target |
|---|---|---|
| A | Exact output | Same next token as the reference on 500 of 500 steps, on a short prompt and again after a 20,000-token prompt |
| B1 | Decode speed, short context | 130 tokens/s or more generating code, 100 or more generating prose, under 1,000 tokens of context |
| B2 | Decode speed at depth | 70 tokens/s or more with 40,000 tokens of context |
| C | Prompt reading | 600 tokens/s or more on a cold 25,000-token prompt |
| D | Cache reuse | When a conversation only grows at the end, re-read no more than the new tokens plus 10% |
| E | Context size | 128,000 tokens fits on the card with at least 1 GiB free |
| F | Serving | OpenAI-style endpoint, bound to localhost, API key required, survives a 15-turn agent session with tool calls |

Targets B1, B2 and C are set above an earlier build on this card, whose own README (commit `fc4b0d8`, read 2026-11-25, numbers not rerun by the brief's author) reports:

| Condition | Earlier build |
|---|---|
| Decode, short context, code | 91.3 tokens/s plain, 117.8 with speculative decoding |
| Decode, short context, second code sample | 90.3 plain, 88.6 with speculative decoding |
| Decode at 32,768 tokens of context | 52.5 plain, 61.8 with speculative decoding |
| Decode at 50,000 tokens of context | 44.2 plain, 50.4 with speculative decoding |
| Prompt reading | 388 tokens/s at 4,096 tokens, 291 at 32,768, 253 at 50,000 |

Speed depends heavily on context depth, so always state the depth with the number. That build's code is not available to you and you should not look for it.

## Rules

1. **Exactness comes first.** No speed number counts unless requirement A passed on the same build.
2. **One change at a time.** Never combine two optimisations in one measurement.
3. **One GPU job at a time.** The card cannot hold two copies of this model. Check nothing else is using it before you run.
4. **Stay in this folder.** Do not read or write outside it, except the paths listed at the top. Do not list or search the home folder, hidden folders, other projects, chat histories or agent memory files, and do not search the web or code hosts for other engines for this model. This is a clean build: the point is to see what you design without seeing anyone else's. If you come across another implementation by accident, stop reading it and tell the human.
5. **Ask before installing anything** that is not on the approved list. Use a virtual environment, never the system Python.
6. **Report what happened.** A failed or slower result is a finding. Write it down as carefully as a success.

## Known pitfall in the reference

llama.cpp's flash attention stores its cache in 16-bit even when asked for 32-bit, and a BF16 GGUF rounds the weights (`measured` 2026-11-21). Either one makes the reference inexact, and you will chase differences that are not your bug. Build the answer key from the unrounded F32 GGUF with flash attention off (`-fa off`). The helper script `~/src/answer-key-f32.sh` does not pass `-fa off` yet (`measured` 2026-11-25), so add that flag before you trust its output. It writes its results to `~/src/bench/key32/`, which you may create and read.

## The loop for every optimisation

Answer these in order and write the answers in `LOG.md` before touching code.

1. **Target:** which requirement (A to F) is this for?
2. **Breakdown:** for one token, how many milliseconds go to each step right now?
3. **Ceiling:** from bytes read per token and the card's memory speed, what is the best possible, and what percent of it are we at?
4. **Limiter:** is the limit memory reading, arithmetic, or overhead between GPU calls?
5. **Prediction:** how much should this one change gain, and why?
6. **Change:** make that one change.
7. **Check:** did the result match the prediction? If it missed by more than a quarter, explain what was misunderstood before going on.
8. **Cost and range:** what did it cost in memory, and does the gain hold at 100, 10,000 and 40,000 tokens of context?

Take every timing as the median of at least five runs after a warm-up, and record the spread.

## Stages

Work through these in order. At the end of each stage, stop, write the stage report, and wait for the human.

| Stage | Build | Gate to pass |
|---|---|---|
| 0 | Read the weight file by hand and write a map of the model: every tensor, its shape, its storage format, and the order of operations for one token | Human can follow the map without reading code |
| 1 | A slow reference on the CPU, written for clarity | Matches the answer key on 20 of 20 next tokens |
| 2 | A single-stream GPU engine, no optimisation yet | Requirement A on a short prompt; first breakdown and ceiling estimate written down |
| 3 | Decode speed at short context, using the loop above | Requirement B1, or 80% of the ceiling, whichever comes first |
| 4 | Long context: decode at depth, prompt reading and cache reuse | Requirements A (long prompt), B2, C, D and E |
| 5 | Serving | Requirement F, then rerun A to E on the served build |

## Stage report

Put this in `LOG.md` at each gate.

- What was built, in five lines or fewer.
- The gate result, with the exact command to rerun it.
- Each number tagged `measured` (you ran it in this stage) or `inferred` (worked out, not run).
- What is not verified.
- **A prediction question for the human about the next stage**, for example "before I measure, which step do you expect to take the most time, and what percent?" Wait for the answer, then measure and compare.

## Stop and ask when

- Requirement A fails and two attempts have not found the cause.
- A measurement comes out above the ceiling you estimated.
- You need a package, a path or a permission you do not have.
- A stage has taken more than three attempts at its gate.
