# Benchmarks

## MC Hand Evaluation (Python vs C++)

```bash
python benchmarks/equity_mc.py --n-samples=10000
```

AA is evaluated against the full range of villains holdings. In total there are 1.68 million hands sampled in this process.

| Language | Total Time  | Speedup (vs Python)|
| -------- | -------- | -------- |
| Row 1 A  | Row 1 B  | Row 1 C  |
| Row 2 A  | Row 2 B  | Row 2 C  |
