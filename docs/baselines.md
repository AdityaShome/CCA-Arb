# Baseline interpretations (GUIDE rule 5)

**FLAG FOR HUMAN: none of the three paper baselines below were implemented from the source
papers.** The papers (DRF, Remember and Reweight, Bayesian Dialectical Argumentation) have not
been read in this repo's development. Each class is a simplest-plausible mechanism matching the
baseline's name and the role the guide gives it (reputation/weights learned WITHOUT counterfactual
outcomes). Before any number from them appears in a paper, replace each with a faithful
implementation or rename it (e.g. "agreement-reputation baseline") and say so.

All baselines learn only from acceptance-side signals (judge scores, agreement with the accepted
action). None sees replayed outcomes. That is the point of the comparison.

| Arbiter | File | What it does here | Fidelity |
|---|---|---|---|
| `majority` | `arbiters/baselines.py` | Plurality over canonical groups; ties by fixed agent order | Exact per guide |
| `acceptance_rep` | same | score = judge score + beta * normalized reputation; reputation +1 on acceptance, geometric decay | Per guide 6.7; the judge sees justifications |
| `drf` | same | Per-agent reputation (EMA of peer support), filter agents below theta, vote weighted by reputation x self_confidence | **Unverified interpretation** |
| `rr` | same | Persistent multiplicative weights, up-weight agents agreeing with the accepted group, down-weight others; weighted vote | **Unverified interpretation** |
| `bda` | same | Posterior over which group is correct from per-agent Beta reliabilities plus judge score; reliabilities updated by agreement with the accepted group | **Unverified interpretation** |
| `oracle` | same | Highest replayed Y (upper bound, replayed steps only; majority fallback otherwise) | Per guide |

## Judge
`HeuristicJudge` (length and confidence) is an offline stand-in that is deliberately persuasion
susceptible. It is for tests and the synthetic simulator only. Headline acceptance-bias results
must use `LLMJudge` with a real model that sees justifications.
