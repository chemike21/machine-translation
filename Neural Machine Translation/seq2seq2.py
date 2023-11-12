#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
from tqdm import tqdm
import numpy as np
import random
import time
from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor

def tensors_from_pairs(src_vocab, tgt_vocab, pairs):
    """creates a tensor from batch of raw sentence pair
    """
    input_tensor_list, target_tensor_list = [], []
    for pair in pairs:
        input_tensor, target_tensor = tensors_from_pair(src_vocab, tgt_vocab, pair)
        input_tensor_list.append(input_tensor)
        target_tensor_list.append(target_tensor)
    input_tensors = torch.nn.utils.rnn.pad_sequence(input_tensor_list, batch_first=True, padding_value=EOS_index)
    target_tensors = torch.nn.utils.rnn.pad_sequence(target_tensor_list, batch_first=True, padding_value=EOS_index)
    return input_tensors, target_tensors


######################################################################


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.wf = nn.Linear(input_size, hidden_size, bias=True)
        self.wi = nn.Linear(input_size, hidden_size, bias=True)
        self.wo = nn.Linear(input_size, hidden_size, bias=True)
        self.wc = nn.Linear(input_size, hidden_size, bias=True)

        self.uf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ui = nn.Linear(hidden_size, hidden_size, bias=False)
        self.uo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.uc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.get_initial_hidden_state()
        h, c = hidden
        x = input
        ft = torch.sigmoid(self.wf(x) + self.uf(h))
        it = torch.sigmoid(self.wi(x) + self.ui(h))
        ot = torch.sigmoid(self.wo(x) + self.uo(h))
        ct = torch.tanh(self.wc(x) + self.uc(h))
        c = ft * c + it * ct
        h = ot * torch.tanh(c)
        return h, c
    
    def get_initial_hidden_state(self):
        return torch.zeros(2, 1, self.hidden_size, device=device).unbind(0)


class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        "*** YOUR CODE HERE ***"
        self.lstm_cell = nn.LSTM(hidden_size, hidden_size, 1, bidirectional=True, batch_first = True)
        self.emb = torch.nn.Embedding(input_size, hidden_size)
        self.fh = nn.Linear(hidden_size*2, hidden_size)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        "*** YOUR CODE HERE ***"
        x = self.emb(input.squeeze(2))
        batch_size = x.size(0)
        hc = self.get_initial_hidden_state(batch_size)
        output, hc = self.lstm_cell(x, hc)
        h_left, c_left = hc[0]
        h_right, c_right = hc[1]
        h = torch.tanh(self.fh(torch.cat((h_left, h_right), dim=1))).unsqueeze(0)
        c = torch.tanh(self.fc(torch.cat((c_left, c_right), dim=1))).unsqueeze(0)
        return output, (h,c)

    def get_initial_hidden_state(self, batch_size):
        return torch.zeros(2, 2, batch_size, self.hidden_size, device=device).unbind(0)

class Highway(nn.Module):
    def __init__(self, size):
        super(Highway, self).__init__()
        self.linear = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        
    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        nonlinear = F.relu(self.linear(x))
        return gate * nonlinear + (1 - gate) * x

class CharacterAwareEncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size, num_output_features=128, kernel_width=3, num_highway_layers=2, dropout_p=0.1):
        super(CharacterAwareEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(num_output_features, hidden_size, 1, bidirectional=True, batch_first = True)
        self.fh = nn.Linear(hidden_size*2, hidden_size)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.emb = torch.nn.Embedding(input_size, hidden_size)
        self.convolution = nn.Conv1d(in_channels=hidden_size, 
                                     out_channels=num_output_features, 
                                     kernel_size=kernel_width)
        
        # Highway layers for combining convolutional features
        self.highway_layers = nn.ModuleList([Highway(num_output_features) for _ in range(num_highway_layers)])

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):
        x = self.emb(input.squeeze(2))
        batch_size = x.size(0)

        x = x.transpose(1, 2)
        x = self.convolution(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        for highway in self.highway_layers:
            x = highway(x)
        x = x.unsqueeze(1)
        
        hc = self.get_initial_hidden_state(batch_size)
        output, hc = self.lstm_cell(x, hc)
        h_left, c_left = hc[0]
        h_right, c_right = hc[1]
        h = torch.tanh(self.fh(torch.cat((h_left, h_right), dim=1))).unsqueeze(0)
        c = torch.tanh(self.fc(torch.cat((c_left, c_right), dim=1))).unsqueeze(0)
        return output, (h,c)

    def get_initial_hidden_state(self, batch_size):
        return torch.zeros(2, 2, batch_size, self.hidden_size, device=device).unbind(0)

class Attention(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        h, _ = hidden
        h = h.repeat(src_len, 1, 1).permute(1,0,2)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.dropout = nn.Dropout(self.dropout_p)
        self.attention = Attention(hidden_size)
        
        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        "*** YOUR CODE HERE ***"

        self.word_emb = torch.nn.Embedding(output_size, hidden_size)
        self.lstm_cell = nn.LSTM(hidden_size*3, hidden_size, 1, bidirectional=False, batch_first = True)
        self.out = nn.Linear(self.hidden_size * 4, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """
        
        "*** YOUR CODE HERE ***"
        attn_weights = self.attention(hidden, encoder_outputs)
        input_enc = self.dropout(self.word_emb(input))
        weighted_encoder = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat([weighted_encoder, input_enc], dim=2)
        _, hc = self.lstm_cell(lstm_input, hidden)
        h, c = hc
        hidden = (h, c)
        out_tensor = self.out(torch.cat((h.permute(1,0,2), weighted_encoder, input_enc), dim = 2))
        log_softmax = out_tensor.log_softmax(2).squeeze(1)
        return log_softmax, hidden, attn_weights


######################################################################

def train(input_tensors, target_tensors, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    "*** YOUR CODE HERE ***"

    optimizer.zero_grad()
    encoder_states, hc = encoder(input_tensors)
    decoder_log_softmax = []
    batch_size = encoder_states.size(0)

    for y in [torch.full((batch_size, 1), SOS_index)] + list(target_tensors.unbind(1))[:-1]:
        log_softmax, hc, attn_weights = decoder(y, hc, encoder_states)
        decoder_log_softmax.append(log_softmax)
    decoder_log_softmax = torch.stack(decoder_log_softmax).permute(1, 0, 2)
    loss = 0
    for i in range(batch_size):
        loss += criterion(decoder_log_softmax[i].unsqueeze(0).flatten(0, 1), target_tensors[i].T.flatten())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss


######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence).unsqueeze(0)
        input_length = input_tensor.size()[1]

        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_index]], device=device)
        

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, input_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    
    "*** YOUR CODE HERE ***"
    attentions = attentions.numpy()
    
    fig, ax = plt.subplots()
    cax = ax.matshow(attentions.T, cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + output_words, rotation=90)
    ax.set_yticklabels([''] + input_sentence.split(' ') + ['<EOS>'])
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.tight_layout()
    plt.show()

    fig.savefig(input_sentence+'.png', bbox_inches='tight')


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=128, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=1000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='translations',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--batch_size', default=1, type=int,
                help='batch size for batch training')
    ap.add_argument('--encoder_mode', default=1, type=int,
                help='which encoder to use, 1 for EncoderRnn and 2 for CharacterAwareEncoderRNN')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)
    if args.encoder_mode == 1:
        encoder = EncoderRNN(src_vocab.n_words, args.hidden_size, dropout_p=0.1).to(device)
    else:
        encoder = CharacterAwareEncoderRNN(src_vocab.n_words, args.hidden_size, dropout_p=0.1).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)
    batch_size = args.batch_size

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    total_iter = args.n_iters
    if args.load_checkpoint is not None:
        total_iter = total_iter - state['iter_num']
    for _ in tqdm(range(total_iter)):
        iter_num += 1
        training_pairs = tensors_from_pairs(src_vocab, tgt_vocab, random.choices(train_pairs, k = batch_size))
        input_tensors = training_pairs[0]
        target_tensors = training_pairs[1]
        loss = train(input_tensors, target_tensors, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            torch.save(encoder.state_dict(), f'encoder.{iter_num}.pkl')
            torch.save(encoder.state_dict(), f'decoder.{iter_num}.pkl')
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
