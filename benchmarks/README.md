# Benchmarks

## Monte-Carlo Equity Evaluation

```bash
python benchmarks/equity_mc.py --n-samples=10000
```

AA is evaluated against the full range of villains holdings. In total there are 1.68 million hands sampled in this process.

| Implementation | Total Time (s)  | Speedup (vs Python)|
| -------- | -------- | -------- |
| Python  | 240.954  | - |
| C++ | 7.097  | 34.0x   |
| CUDA | 1.680 | 143.4x |
