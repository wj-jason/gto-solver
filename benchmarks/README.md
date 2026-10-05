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

## NSight Profiling

Profiling the preflop table build at 2,000 samples per pair showed that the kernel was only about 20% of each call, the rest was host-side allocation, sync, and Python overhead. The kernel itself was also inefficient: a 2,000-trial launch only fills 8 blocks, and scaling to 100,000 trials per launch showed 4.6x higher GPU throughput.

Next step: Batch hand pairs into large launches, possibly dynamically based on `n_samples`. 

```
nsys profile -o equity_2k --trace=cuda,osrt --stats=true --force-overwrite=true python benchmarks/profile_equity.py 20 2000
```

```
nsys profile -o equity_100k --trace=cuda,osrt --stats=true --force-overwrite=true python benchmarks/profile_equity.py 20 100000
```

| Samples per launch | Blocks* | Kernel Time | Trials on GPU** |
| -------------------| ------- | ----------- | --------------- |
| 2000 | 8 | 262us | ~7.6 million | 
| 100000 | 391 | 2870us | ~35 million | 


\* *Blocks = n_samples // 256* 

\*\* *Trials on GPU = n_samples / kernel time per launch* 