# Run 3 (planned before it ran, 2026-11-25)

Pair reversed: source egress-vetting, target command-vetting (run 2's no-memory arms improved egress from 0.67 to 0.93-0.97 evolve, so a source run can plausibly
promote; the command task is not improvable by prompt rewriting with this model). Same configuration as run 2 (feedback v2, record_rejected, 6 rounds, 2 candidates,
temperature 0.3 for the engineers, 0 and 400 tokens for the policy) so that the pair is the only change. Held-out outside every loop. What would count as a result:
the source run promotes at least one edit and its memory is non-empty; then the four target arms differ in start harness (cold start from seed, the other three from the
evolved harness) and memory, which is what the paper compares. If the source run again accepts nothing, run 3 is recorded as another negative result on the design.
