## Command Line Argument
```./kernel_example [-m] [-p] [SampleSize]```
- XORWOW is default generator.
- -m: To use MRG32k3a generator
- -p: To use Philox4_32_10_t generator
- SampleSize: To customize the size of sample, default is 10000.



## Efficient Result
As per the experiments, By using 128 blocks, each having 256 threads, we can get the reliable outcome.
Some snippets of Running time seen-

| Blocks | Threads | _real_ | _user_ | _sys_ |
| ------ | ------- | ------ | ------ | ----- | 
| 64 | 64 | 0m0.347s | 0m0.094s | 0m0.229s |
| 128 | 256 | 0m0.443s | 0m0.193s | 0m0.230s |
| 256 | 64 | 0m0.286s | 0m0.094s | 0m0.169s |
