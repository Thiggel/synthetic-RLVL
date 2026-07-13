# Qwen2.5 Nanotron HF RoPE Compatibility Audit

Last updated: 2026-07-13 17:56 CEST.

## Finding

Nanotron trained Qwen2.5 with the intended `rope_theta=1000000`. Conversion ran
under Transformers 5.12.1, which serialized that value as:

```json
"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}
```

Downstream SFT/evaluation runs under Transformers 4.57.3. That version ignores
`rope_parameters` for Qwen2 and silently resolved the absent legacy
`rope_theta` field to `10000`. Model weights and Nanotron optimizer/checkpoint
state are unaffected; HF inference and the two completed UltraChat adapters are
affected.

## Remediation

- Converted configs now include both `rope_parameters.rope_theta` and legacy
  top-level `rope_theta`, with equal values.
- The downstream-environment verifier rejects a staged upload unless both
  fields equal `1000000` and `AutoConfig` under Transformers 4.57 resolves the
  same value.
- Control and corrected-NL Hub configs were repaired in place and independently
  verified through Transformers 4.57.
- The two affected local and Hub instruction adapters were deleted and are
  being retrained as jobs `3850351/3850352`.
- All previous control/NL direct and instruction benchmark bundles, including
  post-hoc MATH-500 symbolic scores, are quarantined until rerun. Pending logic
  conversion uses the fixed converter and verifier.
- The four invalid bundle directories are preserved with
  `.rope10000_invalid_20260713` suffixes. Corrected direct reviewer-suite jobs
  are `3850385/3850386`, corrected instruction jobs are `3850387/3850388`,
  and corrected six-way aggregate `3850389` supersedes canceled `3849776`.

## Multi-hop context audit

The first HotpotQA/2Wiki/MuSiQue smoke additionally used `max_model_len=8192`.
lm-eval explicitly reported left truncation for prompts between 11.8k and
17.3k tokens. Complete Qwen-tokenizer measurement over all 600 LongBench
examples found maxima 17684, 17079, and 17927 tokens respectively. Corrected
evaluation therefore uses and audits a 32768 window, which retains every
prompt plus the 512-token tagged generation allowance.
