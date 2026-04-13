# Ablation Study: <study-name>

## Question
<!-- "Which component/parameter matters most for X?" -->


## Base configuration
<!-- The full system — all components active, default parameters -->
Config: 
Model: 
Dataset: 

## Variable matrix

<!-- Start with single-variable ablations. Add interactions only when results suggest they matter. -->

| Run | Variable A | Variable B | Variable C | Expected effect |
|-----|-----------|-----------|-----------|-----------------|
| base | default | default | default | baseline |
| -A | **changed** | default | default | isolate A's contribution |
| -B | default | **changed** | default | isolate B's contribution |
| -C | default | default | **changed** | isolate C's contribution |

## Common ablation dimensions for RL agent training

<!-- Pick the ones relevant to your study. Delete the rest. -->

### Reward design
| Variant | Reward function | Expected effect |
|---------|----------------|-----------------|
| F1 (base) | root_cause_f1_reward | balanced precision/recall |
| exact match | exact set match (0/1) | sparser signal |
| partial + bonus | F1 + bonus for full match | stronger gradient at top |

### Training dynamics
| Variant | Parameter | Values |
|---------|-----------|--------|
| KL sweep | kl_ctl | 0.0, 0.01, 0.05, 0.1 |
| LR sweep | lr | 1e-6, 5e-6, 1e-5, 5e-5 |
| Staleness | max_head_offpolicyness | 0, 2, 4, 8 |
| Clip range | eps_clip | 0.1, 0.2, 0.4 |

### Agent workflow
| Variant | Parameter | Values |
|---------|-----------|--------|
| Turn discount | turn_discount | 0.8, 0.9, 0.95, 1.0 |
| Export style | export_style | individual, concat |
| Samples per prompt | n_samples | 1, 4, 8, 16 |

### Model scale
| Variant | Model | Size |
|---------|-------|------|
| small | Qwen2.5-0.5B-Instruct | 0.5B |
| medium | Qwen2.5-7B-Instruct | 7B |
| large | Qwen2.5-14B-Instruct | 14B |

## Execution order
<!-- Run baselines first, then single-variable ablations, then interactions if needed -->
1. 
2. 
3. 
