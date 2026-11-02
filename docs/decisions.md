# Design decisions and known caveats

1. **Session = env + agent-visible history.** Checkpoints include the conversation so a restored
   branch shows agents the right context (`cca/session.py`).
2. **Y after episode end.** tau-bench's `step()` runs `calculate_reward()` on done, which overwrites
   `env.data`. `Session.outcome()` therefore uses the reward returned by the final step when done,
   and `env.verify()` (run on a deep copy with a null user) only when not done.
3. **Unique-branch replay.** Proposals with the same canonical form share one replayed branch;
   cost counts unique branches. Agents in the same group get the same Y.
4. **Replay happens before select, but only the Oracle may see its outcomes.** All other arbiters
   select from context that excludes this step's outcomes, then learn from them.
5. **Paired continuation seeds.** Continuation k uses the same seed across branches so branch
   differences come from the first action.
6. **Delayed credit.** On non-replayed steps the accepted agent gets the episode reward at weight
   `w_delayed`, applied at episode end (`flush_episode`).
7. **Learned trigger explores** with probability `explore` so skipped regions keep getting labels
   (otherwise selection bias freezes the model).
8. **In-process replay.** Branches run in the same process via restore. Guide 6.5 asks for isolated
   processes/containers; this is fine for tau-bench's pure-Python DB but must be revisited for
   SWE-bench (M6).

## Caveats
- User-simulator tokens/cost are not yet counted in `cost.py`.
- GPU-hours are summed LLM-call latency, an upper bound if calls overlapped.
- `MockAgent` and the synthetic simulator test plumbing and algorithm behaviour under assumed
  generative models. Nothing from them is an experimental result.
- Replay horizon cap `H` truncates; truncated branches are scored by `verify()` on partial state.
