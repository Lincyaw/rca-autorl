# RL reward, worked through one case

What a rollout earns, why it earns that, and where each number comes from. The
case, its annotation, and the submission below are real — `case330`,
`ts1-ts-route-plan-service-stress-pvnmb`, from the fifty collected episodes. The
seven siblings are constructed, because a group of eight rollouts of one case
has not been collected yet; every number derived from them is reproducible from
`autorl.difficulty`.

## 1. The episode

The model is given the incident and three tools — `sql` over the snapshot,
`take_note`, and the terminal `submit_result`. This episode ran 93 tool calls
before submitting. Each LLM request becomes one training sample
(`export_style: individual`), so the episode is ~45 samples, not one.

## 2. The annotation

`datapacks/ops-lite/cases/ts1-ts-route-plan-service-stress-pvnmb/causal_graph_verified.json`:

| element | role |
| --- | --- |
| `svc:ts-route-plan-service` | root cause — the injected service |
| `svc:ts-travel-plan-service` | the middle of the chain |
| `svc:ts-ui-dashboard` | the symptom, and **named in the incident text** |

The third one matters. The incident says the SLO violations are on
`http://ts-ui-dashboard:8080/...`, so an episode that filters on the endpoint it
was handed already "found" a node of the true graph.

## 3. The submission

The model named nine services, which include all three true ones:

```
ts-basic-service   ts-order-service   ts-route-plan-service   ts-seat-service
ts-station-service ts-travel-plan-service ts-travel-service   ts-travel2-service
ts-ui-dashboard
```

Perfect recall, poor precision. `fpg.compare_model_to_ground_truth` scores it
**0.225**.

## 4. The group

Eight rollouts of the same case. One is the real submission; the rest are the
two failure shapes the collected episodes actually show — stopping at the middle
of the chain, and stopping at the symptom.

| # | found | claimed |
| --- | --- | --- |
| ① ×1 | route-plan, travel-plan, ui-dashboard | 9 services |
| ② ×3 | travel-plan, ui-dashboard | 3 services |
| ③ ×4 | ui-dashboard | 2 services |

## 5. Difficulty, read off the columns

For each true element, how many of the eight found it. The weight is `1 − found/8`.

| element | found by | weight |
| --- | --- | --- |
| `svc:ts-ui-dashboard` | 8/8 | **0.000** |
| `svc:ts-travel-plan-service` | 4/8 | 0.500 |
| `svc:ts-route-plan-service` | 1/8 | **0.875** |

The symptom the incident named is worth nothing, because finding it separated
nobody. The injected service is worth almost everything, because one rollout
found it. No external label was needed for either.

## 6. The scores

Recall is weighted by those numbers; precision is not, because difficulty is a
property of the truth and a claim outside the graph is wrong wherever it lands.
Per axis, `F1 = 2pr/(p+r)`, combined `0.4·roots + 0.3·subjects + 0.3·edges`.

| | flat | weighted |
| --- | --- | --- |
| ① real submission | 0.225 | **0.5500** |
| ② missed the root | — | 0.1412 |
| ③ symptom only | — | **0.0000** |

③ scores exactly zero. Everything it found was free.

## 7. Advantage

`GroupedRolloutWorkflow` merges the eight into one trajectory;
`reward_norm(mean_level="group", mean_leave1out=true)` centres each turn against
every turn of every sibling. Because every turn of an episode carries the same
value, that baseline is the mean weighted score of the siblings:

```
advantage(①) = 0.5500 − mean(others) = 0.5500 − 0.0605 = +0.4895
advantage(③) = 0.0000 − mean(others) = 0.0000 − 0.1391 = −0.1391
```

Every one of ①'s ~45 turns carries `+0.4895`. There is no turn-level credit:
the update says "more of what this episode did", uniformly.

## 8. What it costs to be conservative

The incentive this weighting exists to fix. Take two rollouts that name the same
*number* of true services, one of which spent its budget on the root instead of
on an easy neighbour. Over the 28 collected cases with at least two easy
entities and a root:

| | ambitious beats conservative | mean gap |
| --- | --- | --- |
| flat | 0/28 (all ties) | +0.000 |
| weighted | **28/28** | +0.190 |

Under a flat score the two are worth the same, so the cheaper strategy wins on
cost: sweep the easy entities, never attempt the expensive one. That is a
policy the flat reward would actively train.

## 9. What is deliberately absent

**No process supervision.** Nothing judges a step. A per-block signal — the
share of a block's queries that filtered on a true entity — was built and
removed: it measures which SQL the agent ran, which is its strategy rather than
its output, and decomposing its 0.35 correlation with the outcome showed the
strongest part (0.41) was querying the service the incident text already names,
against 0.24 for root causes and 0.14 for the middle of the chain. Within a
chaos family it ranged 0.07 to 0.67 over six to fifteen episodes. `shaping`
defaults to 0.0 and the code path remains for the next candidate.

**No critic.** `discount` and `gae_lambda` are both 1 and `critic` is null, so
GAE is an identity and the algorithm is RLOO. The PPO clip that remains is
off-policy correction for asynchronous rollout, not step-size control.

**No turn-level credit.** Section 7. The citation channel that could have
provided it does not: only 12.6% of the SQL statements a submission cites match
a query the episode actually ran — the model composes a clean statement after
the fact rather than quoting one.

## 10. What is not yet known

**The real within-group variance.** Everything above turns on the eight rollouts
differing. The one case observed more than once (`case0`, three rollouts) had
all three find the same four entities — weights all zero, advantage zero. If
that is common the design is sound and inert. It can only be measured by running
several rollouts of one case, which has not been done.

**Whether the path runs at all.** Two seams have never executed end to end: the
rollout proxy reached as a declared `llm-pi-ai` route (verified only against a
recorded stream), and `rescore_group` (verified only in unit tests).
