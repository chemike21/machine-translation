There are three python programs here (-h for usage):

- `decode` translates input sentences from French to English.
- `decode-ext` the decoder we implemented, support four modes: simple decoder, beam-search decoder, greedy decoder, greedy decoder from beam-search.
- `grade` computes the model score of a translated sentence.

These commands work in a pipeline. For example:

    > python decode | python grade

To choose mode for `decode-ext`, use -m argument, 0 for simple decoder, 1 for beam search, 2 for greedy, 3 for greedy from beam search. The default mode is greedy decoder from beam-search.

For example:
    > `python3 decode-new -m 3 -s 10000 > translations`

There is also a module:

- `model.py` implements very simple interfaces for language models
 and translation models, so you don't have to. 

You can finish the assignment without modifying this file at all. 
You should look at it if you need to understand the interface
to the translation and language model.

The `data` directory contains files derived from the Canadian Hansards,
originally aligned by Ulrich Germann:

- `input`: French sentences to translate.

- `tm`: a phrase-based translation model. Each line is in the form:

    French phrase ||| English phrase ||| log_10(translation_prob)

- `lm`: a trigram language model file in ARPA format.

    log_10(ngram_prob)   ngram   log_10(backoff_prob)

The language model and translation model are computed from the data 
in the align directory, using alignments from the Berkeley aligner.
