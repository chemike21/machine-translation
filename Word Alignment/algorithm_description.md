Group member name: Jiayuan Hu, Keyi Lu, Yu Che

**Implemented solution**

We first Implement the IBM Model 1. This gives us a result of AER = 0.38.

Then we trained a symmetric model of Model 1 and output the joint set of results. By carefully adjusting parameters to threshold = 0.25, num_iterates = 20, and num_sentences = 10000, we got an alignment result with AER = 0.27.

