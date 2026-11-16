# Lost in Transliteration

*Tiny Aya knows romanized Hindi and Punjabi requests are harmful, but doesn't refuse them. Where it breaks, and two fixes.*

**Team LASTROUNDCLUTCH** · Sunit Singh Dhaliwal & Ramit Joshi · Lexsi Labs × Cohere Labs × Hugging Face AI Safety and Alignment Hackathon · Tracks 2b (#9 locating safety, #8 steering) and 2a (#4 multilingual & code-mixed safety)

[**📄 Article (HF)**](https://huggingface.co/blog/sunit12/lost-in-transliteration) · [**▶ Reproduce in 1 min (Colab)**](https://colab.research.google.com/github/SunitSinghDhaliwal/lost-in-transliteration/blob/main/reproduce.ipynb) · [**🔍 Explore every output**](https://sunit12-knows-but-doesnt-refuse-explorer.static.hf.space) · [Models](https://huggingface.co/collections/sunit12/knows-but-doesnt-refuse-models-6af82fe3ced10d8fb2f69f91) · [Data & logs](https://huggingface.co/collections/sunit12/knows-but-doesnt-refuse-data-and-logs-6af82fe439fca752611786f2)

> ⚠️ Content warning: the benchmark and logs quote harmful requests and some unguarded model outputs.

**Claim.** Tiny Aya *recognises* harmful requests typed in romanized Hindi and Punjabi (an English-trained harm probe still fires, AUROC 0.87–0.95), but the refusal direction never switches on. Adding back exactly the refusal component the English twin has, and nothing else, restores refusal. **CircuitKIT** shows the refusal circuit's overlap with English falling from 0.60 to 0.21 exactly where refusal fails. Repairing it at those layers works twice:

![Headline figure](results/fig_headline_v4.png)

| Held-out, judge-scored | Plain Tiny Aya | Fix 1: gated steer (no training) | **Fix 2: circuit-guided LoRA** |
|---|---|---|---|
| Usable refusal, non-English harmful (184) | 38% | 76% (+38, CI +29 to +47) | **90%** (+52, CI +43 to +60) |
| … romanized Hindi / romanized Punjabi | 26% / 15% | 63% / 63% | **91% / 87%** |
| Four unseen romanized languages | 15% | 57% | **78%** |
| Non-English harmless prompts refused | 13.5% | 15.1% | 17.7% (+4, CI −1 to +9) |
| MMLU (400 q) | 50.5% | 50.5% | **50.5%** |

*Usable refusal = 4-way LLM judge says REFUSAL, the reply is not a degenerate loop, and it is in a language the user can read. SafeTune CAST on the same prompts: 40%. Judge vs native speaker: κ = 0.63 on 29 items, drawn before Fix 2 existed. A 201-item blinded check of all three runs and a second judge from another model family are ready to run ([`validation/README.md`](validation/README.md)). MMLU uses 400 questions, so it can only rule out drops larger than about 5 points. Fix 2's over-refusal, prompt by prompt: [`results/overrefusal_lora.md`](results/overrefusal_lora.md).*
*Post-submission check: 159 of the LoRA's 165 usable refusals on non-English harmful prompts (96%) contain explicit refusal wording; counting the 2 clear judge errors among the rest gives 89% instead of 90% ([details](results/refusal_wording_check.md)).*


## Quick start

**1. Check any model for this failure** (generate → 4-way judge → language-ID → usable refusal):
```bash
pip install -r requirements.txt
python codemix_eval.py --model <any HF chat model> --limit 5          # 2-min smoke test
python codemix_eval.py --model CohereLabs/tiny-aya-global --adapter sunit12/tiny-aya-global-codemix-refusal-lora
```
**2. Use the fix** (Fix 2, no hooks):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("sunit12/tiny-aya-global-codemix-refusal-lora")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "sunit12/tiny-aya-global-codemix-refusal-lora").merge_and_unload()
```
On CPU with llama.cpp: `llama-cli -m tiny-aya-global-q4_k_m.gguf --lora codemix-refusal-lora-f16.gguf` (the adapter is in the LoRA repo). Fix 1 (gated steer, 9 KB): `guard.py` in [`sunit12/tiny-aya-global-codemix-refusal-guard`](https://huggingface.co/sunit12/tiny-aya-global-codemix-refusal-guard).

**3. Reproduce the numbers:** [`reproduce.ipynb`](reproduce.ipynb) recomputes the table above, its CIs and the figure from the public logs on a CPU in about a minute, then runs base vs LoRA live on a GPU.

## Repo layout

| Path | What |
|---|---|
| [`reproduce.ipynb`](reproduce.ipynb) | Reproduce the headline (CPU), live demo, full benchmark run (GPU) |
| [`codemix_eval.py`](codemix_eval.py) | The benchmark as a one-command tool for any HF model (also importable: `summary`, `paired_bootstrap`, …) |
| [`docs/hf-article.md`](docs/hf-article.md) | **Full write-up**, organised by the hackathon's 9-point checklist |
| [`docs/archive/`](docs/archive/) | Earlier article versions, drafts and working notes |
| `scripts/jobs/` | Every GPU experiment (`hf jobs uv run scripts/jobs/<job>.py …`): baseline + hidden states, judge, steering, CAST, causal patching, CircuitKIT, AlignTune distillation, MMLU |
| `scripts/analysis/`, `scripts/gate/`, `scripts/data/`, `scripts/pipelines/` | Layer analysis, tables + bootstrap CIs, figures, gate training, distillation-data builders, end-to-end pipelines |
| `results/` | Every table (`.md` / `.json`) and figure in the article |
| `data/`, `validation/` | Test sets and the native-speaker judge validation |
| `auditkit_codemix/` | AuditKIT scenario + `MultilingualRefusalDetector` (AuditKIT's built-in detector has 0% recall on Hindi/Punjabi) |
| `publish/`, `space/`, `space_demo/` | Model cards and guard code as released on HF; the output explorer; a llama.cpp base-vs-LoRA demo (Docker) |
| `artifacts/gates/` | Trained gate weights |

Every script declares its dependencies inline (PEP 723), so `uv run` / `hf jobs uv run` needs no setup. Run scripts from the repo root. Kits used: **CuratorKIT, SafeTune, AuditKIT, CircuitKIT, AlignTune**. Models: Tiny Aya Global and Fire, Aya Expanse 8B (judge). Whole project ≈ $18 of GPU.

**Team.** Sunit Singh Dhaliwal ([@sunit12](https://huggingface.co/sunit12), project lead: question, benchmark and native-speaker review, mechanism, both fixes) and Ramit Joshi ([@orionfax](https://huggingface.co/orionfax), related work, critical review, independent testing, figures, deck and submission). Contributions in full in the [article](docs/hf-article.md#9-team).

Tags: `v4.2-submission` (this version) · `v4.1-submission` · `v1.0-submission`. Models and data inherit CC-BY-NC-4.0 (Tiny Aya) and the dataset licences listed in the article.
