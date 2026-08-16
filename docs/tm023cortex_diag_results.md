# TM.0.23.CORTEX.DIAG results

**product:** `0.0.004`
**earned_next:** `False`
**ex0s:** `None`
**trace_purity_ok:** `True`
**neural_mechanism_changed:** `False`

## CPU D1-like summary (pair 0 main)

- op_counts: `{'STOP': 12, 'EMIT': 1, 'ACT': 59, 'HOLD': 47, 'WRITE': 1}`
- act_rate: `0.49166666666666664` hold_rate: `0.39166666666666666`
- forced_hold_suspect_rate: `0.0`
- vocab_ever_has_press: `True` harm: `True`
- act_token_counts: `{'press': 59}`
- pred_err early→late: `3.5585519991070877` → `1.3976676503797125`
- mean_adv: `-0.035` frac_adv_pos: `0.025`
- prob_gain: `{'delta_p_act': -0.09355122703898347, 'p_act_birth': 0.3187683139430674, 'p_act_end': 0.22521708690408396, 'p_hold_birth': 0.13624633721138651, 'p_hold_end': 0.0996316253626817}`
- credit: `{'STOP': {'mean': 0.0, 'frac_pos': 0.0, 'n': 12}, 'EMIT': {'mean': -1.0, 'frac_pos': 0.0, 'n': 1}, 'ACT': {'mean': -0.03728813559322033, 'frac_pos': 0.05084745762711865, 'n': 59}, 'HOLD': {'mean': 0.0, 'frac_pos': 0.0, 'n': 46}, 'WRITE': {'mean': -1.0, 'frac_pos': 0.0, 'n': 1}}`

cpu_gpu_divergence: `{'ok': True, 'cpu_op_counts': {'STOP': 12, 'EMIT': 1, 'ACT': 59, 'HOLD': 47, 'WRITE': 1}, 'gpu_op_counts': {'STOP': 12, 'EMIT': 1, 'ACT': 59, 'HOLD': 47, 'WRITE': 1}, 'policy': 'HOLD/ACT count parity on identical seed D1-like trajectory'}`

Raw traces: gitignored `runs/cortex_diag/` (SHAs in lock).
