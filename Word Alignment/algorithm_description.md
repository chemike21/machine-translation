**Implemented solution**

To align words, run program `./align` (`-h` for usage).
For example:
> ./align -n 10000 -t 0.25 -i 20 > alignment

Implement the IBM Model 1. This leads to a result of AER = 0.38.

Then train a symmetric model of Model 1 and output the joint set of results. By adjusting parameters to threshold = 0.25, num_iterates = 20, and num_sentences = 10000, finally get an alignment result with AER = 0.27.

