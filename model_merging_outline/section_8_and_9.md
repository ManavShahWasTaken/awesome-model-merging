# Section 8 & 9: What Works at Scale + Practical Guidance

---

## Slide 1: The Scaling Wall

**Merging methods and LLMs.**

- A lot of the methods we discussed were developed on computer vision models and some early language models. It helps build an intuition for how merging works, but not all the results transfer to LLMs.
- TIES-Merging pushes weights L2 distance 100–300+ from base → catastrophic degradation
- Subspace methods (TSV-Merge, Iso-C, Subspace Boosting) show poor results on LLMs
- But as we scale to larger models, the loss basin is flat enough that most methods work well.

**Why?** Vision models have near-orthogonal task structure from contrastive pretraining. LLM fine-tunes don't — task vectors are entangled and overlapping.

> Source: "A Systematic Study of Model Merging Techniques in LLMs" (arxiv 2511.21437) — tested 6 methods across Llama 3.2 3B, Llama 3.1 8B, Qwen3 4B/8B on 16 benchmarks

**Speaker notes:** "If you remember one thing from this section — is to look out for more research on LLMs."

---

## Slide 2: Bigger Models Merge Better

**At scale, method choice vanishes.**

|  | **1B** | **64B** |
|---|---|---|
| Method variance | High (TIES > Avg) | Near-zero (all ≈ equal) |
| Base model quality impact | Moderate | **Dominant** |
| # experts before degradation | ~4 | 8+ |

**The base model is the single biggest lever:**
- Pretrained base, 8 experts merged: 0.66 → 0.39 (destroyed)
- Instruction-tuned base, 8 experts merged: 0.91 → 0.86 (preserved)

At scale, merged models generalize better to held-out tasks than individual experts — and can outperform multitask-trained models.

> Source: Yadav et al., "What Matters for Model Merging at Scale?" (TMLR 2025) — PaLM-2 at 1B/8B/24B/64B + Llama-2 at 7B/13B/70B

**Speaker notes:** "The practical implication: spending time on finding the best instruction-tuned base is very important and will be the bulk of where your performance comes from."

---

## Slide 3: MergeBench Rankings

**The best method depends on your data budget.**

| Data available | Best method | Runner-up |
|---|---|---|
| **Zero data** | Model Soup (simple avg) | Task Arithmetic |
| **Unlabeled validation** | Task Arithmetic | Localize-and-Stitch |
| **Labeled training data** | RegMean | Localize-and-Stitch |

**Consistently low-tier:**
- DARE — dropout randomness hurts reliability
- Fisher Merging — diagonal approximation insufficient at scale

> Source: MergeBench (NeurIPS 2025) — 8 methods, Llama-3.2-3B/Llama-3.1-8B/Gemma-2-2B/9B, 5 domains

**Speaker notes:** "Most of you have at least some unlabeled data. Default to Task Arithmetic."

---

## Slide 4: What Frontier Labs Actually Do

**Two paradigms at 100B+**

### Paradigm 1: Checkpoint averaging during pretraining
- **Meta Llama 3.1 405B** — Polyak averaging during annealing phase
- **ByteDance PMA on 70B dense** — HumanEval: 50.6 → 57.9, GSM8K: 85.9 → 91.3
- Validated up to **200B MoE** architectures

### Paradigm 2: Post-training merge of RL specialists
- **GLM-4.5** (355B MoE) — merges instruction + reasoning → 91.0% AIME 2024
- **DeepSeek-V3** (671B MoE) — uses merging during development

**Caveat:** Diminished returns when models have converged to very low LR — checkpoints cluster too tightly in the loss basin.

**Speaker notes:** "Note the gap between these two paradigms — checkpoint averaging during pretraining vs post-hoc merging of independent fine-tunes. The community mostly does the latter; frontier labs mostly do the former. Bridging this gap is where the most impactful research remains."

---

## Slide 5: The Practitioner's Decision Tree

**What to use when**

```
Merging 2 models?
  └─→ SLERP with layer-wise gradient t values
       (inverse gradients for self_attn vs MLP across depth)

Merging 3+ models?
  └─→ Task Arithmetic (λ = 0.5 for LLMs)
       or simple averaging

Have unlabeled validation data?
  └─→ AdaMerging (layer-wise)
       or mergekit-evolve

Have compute for search?
  └─→ Evolutionary optimization (CMA-ES)
```

**The universal rule:** Invest more time selecting high-quality instruction-tuned source models than optimizing merging hyperparameters.

**Speaker notes:** "This is the slide to photograph. For 90% of use cases, SLERP for pairwise and Task Arithmetic for multi-model is all you need."

---

## Slide 6: Recipes That Work

**Validated hyperparameter configurations**

| Method | Key param | Default | Proven range / notes |
|---|---|---|---|
| **SLERP** | t | 0.5 | Gradient per layer: `[0, 0.5, 0.3, 0.7, 1]`. Use inverse gradients for self_attn vs MLP |
| **TIES** | density | 0.5 | Range: 0.2–0.6. λ = 0.5 for LLMs, 0.3 for vision |
| **DARE** | density | 0.53 | Range: 0.3–0.7. Weights sum to 0.9–1.1. Use `int8_mask: true` |
| **Task Arith** | λ | 0.5 | Math: 0.5–0.6 · Code: 0.1–0.2 · Chat: 0.2–0.3 |

**Critical constraints:**
- DARE only works when delta magnitudes are small (~0.005) — typical of SFT, fails for continued pretraining
- SLERP is limited to 2 models — chain hierarchically for 3+
- Cosine similarity ≥ 0.98 between source model weights predicts stable merging

**Speaker notes:** "The SLERP gradient trick is what made Marcoro14-7B-slerp the #1 7B model on the leaderboard. Early layers favor one model's attention but the other's MLP, then cross over. This works because self-attention carries relational reasoning while MLP layers carry factual knowledge."

---

## Slide 7: How to Not Fool Yourself

**Evaluation & contamination**

**Rule 1: Never evaluate on one benchmark.**
- Minimum: Open LLM Leaderboard tasks (ARC, HellaSwag, MMLU, Winogrande, GSM8K, TruthfulQA) + MT-Bench
- Better: add NousResearch suite (AGIEval, GPT4ALL, BigBench) + LiveBench

**Rule 2: Merged models inherit contamination from all parents.**
- GSM1k studies found up to 8% accuracy drops on clean test versions
- Use dynamic benchmarks: LiveBench, LiveCodeBench
- Flag contamination risk honestly in model cards

**Rule 3: Post-merge, use DPO — not SFT.**
- SFT shows no improvement on already-SFT'd merged models
- DPO yields 1–3% gains over the merged model

**Speaker notes:** "Labonne himself warns: 'By merging the best models, we also contaminate our own results.' This is a systemic risk the community has not adequately addressed."

---

## Slide 8 (Optional): Hardware Is Surprisingly Cheap

**You can merge 70B models on your laptop.**

MergeKit uses lazy tensor loading — never loads all models simultaneously.

| Scale | RAM needed | GPU needed | Time |
|---|---|---|---|
| **7B merge** | 16–64 GB | None | Minutes |
| **70B merge** | 64–128 GB | None | Hours |
| **LazyMergeKit (Colab)** | 13 GB (free tier) | None | ~30 min for 7B |

**Key flags:**
```bash
mergekit-yaml config.yaml output/ \
  --lazy-unpickle \
  --out-shard-size 1B \
  --low-cpu-memory
```

Disk: ~500 GB for two 70B source models + output.

**Speaker notes:** "The barrier to entry is essentially zero. If you have a MacBook with 64GB RAM, you can merge 70B models tonight."
