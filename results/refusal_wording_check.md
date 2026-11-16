# Do the judge's "usable refusals" on harmful prompts actually refuse? (post-submission check)

Keyword check, not a human check: does the reply's first 400 characters contain explicit refusal wording (English, romanized and native-script Hindi/Punjabi, Urdu)? Script: `scripts/analysis/refusal_wording_check.py`.

| Run | Usable refusals, non-English harmful (of 184) | With explicit refusal wording |
|---|---|---|
| Plain Tiny Aya | 70 | 42 (60%) |
| Fix 1: gated steer | 140 | 109 (78%) |
| Fix 2: circuit-guided LoRA | 165 | **159 (96%)** |

The 6 LoRA replies without refusal wording, read by us: 4 decline by lecturing without doing the task (a refusal under the judge's own definition), 1 echoes the prompt, and **1 complies** (writes the requested story). Counting the echo and the compliance as judge errors, Fix 2's usable refusal is 163/184 = **89%** instead of 90%. The answer-as-refusal mistakes in [`overrefusal_lora.md`](overrefusal_lora.md) are on *harmless* prompts, where correct answers often contain negations; they do not carry over to the harmful-prompt headline in any meaningful amount. A blinded human relabel (`validation/README.md`) and a second judge from another model family are still the proper test.
