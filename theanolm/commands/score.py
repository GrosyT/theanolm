#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import numpy
import theano
import theanolm
from filetypes import TextFileType

def add_arguments(parser):
    argument_group = parser.add_argument_group("files")
    argument_group.add_argument(
        'model_path', metavar='MODEL', type=str,
        help='path where the best model state will be saved in numpy .npz '
             'format')
    argument_group.add_argument(
        'input_file', metavar='INPUT', type=TextFileType('r'),
        help='text or .gz file containing text to be scored (one sentence per '
             'line)')
    argument_group.add_argument(
        'dictionary_file', metavar='DICTIONARY', type=TextFileType('r'),
        help='text or .gz file containing word list or class definitions')
    argument_group.add_argument(
        '--dictionary-format', metavar='FORMAT', type=str, default='words',
        help='dictionary format, one of "words" (one word per line, default), '
             '"classes" (word and class ID per line), "srilm-classes" (class '
             'name, membership probability, and word per line)')
    argument_group.add_argument(
        '--output-file', metavar='OUTPUT', type=TextFileType('w'), default='-',
        help='where to write the score or rescored n-best list (default '
             'stdout)')
    
    argument_group = parser.add_argument_group("scoring")
    argument_group.add_argument(
        '--output', metavar='DETAIL', type=str, default='text',
        help='what to output, one of "perplexity", "utterance-scores", '
             '"word-scores" (default "perplexity")')
    argument_group.add_argument(
        '--log-base', metavar='B', type=int, default=None,
        help='convert output log probabilities to base B (default is the '
             'natural logarithm)')

def score(args):
    print("Reading model state from %s." % args.model_path)
    state = numpy.load(args.model_path)
    
    print("Reading dictionary.")
    sys.stdout.flush()
    dictionary = theanolm.Dictionary(args.dictionary_file, args.dictionary_format)
    print("Number of words in vocabulary:", dictionary.num_words())
    print("Number of word classes:", dictionary.num_classes())
    
    print("Building neural network.")
    sys.stdout.flush()
    architecture = theanolm.Network.Architecture.from_state(state)
    print(architecture)
    network = theanolm.Network(dictionary, architecture)
    print("Restoring neural network state.")
    network.set_state(state)
    
    print("Building text scorer.")
    sys.stdout.flush()
    scorer = theanolm.TextScorer(network)
    
    print("Scoring text.")
    if args.output == 'perplexity':
        _score_text(args.input_file, dictionary, scorer, args.output_file,
                    args.log_base, False)
    elif args.output == 'word-scores':
        _score_text(args.input_file, dictionary, scorer, args.output_file,
                    args.log_base, True)
    elif args.output == 'utterance-scores':
        _score_utterances(args.input_file, dictionary, scorer, args.output_file,
                          args.log_base)

def _score_text(input_file, dictionary, scorer, output_file,
                log_base=None, word_level=False):
    """Reads text from ``input_file``, computes perplexity using
    ``scorer``, and writes to ``output_file``.

    :type input_file: file object
    :param input_file: a file that contains the input sentences in SRILM n-best
                       format

    :type dictionary: Dictionary
    :param dictionary: dictionary that provides mapping between words and word
                       IDs

    :type scorer: TextScorer
    :param scorer: a text scorer for rescoring the input sentences

    :type output_file: file object
    :param output_file: a file where to write the output n-best list in SRILM
                        format

    :type log_base: int
    :param log_base: if set to other than None, convert log probabilities to
                     this base

    :type word_level: bool
    :param word_level: if set to True, also writes word-level statistics
    """

    validation_iter = theanolm.BatchIterator(input_file, dictionary)

    total_logprob = 0
    num_words = 0
    num_sentences = 0
    for word_ids, membership_probs, mask in validation_iter:
        logprobs = scorer.score_batch(word_ids, membership_probs, mask)
        for seq_index, seq_logprobs in enumerate(logprobs):
            seq_logprob = sum(seq_logprobs)
            seq_length = len(seq_logprobs)
            total_logprob += seq_logprob
            num_words += seq_length
            num_sentences += 1
            if not word_level:
                continue
            seq_word_ids = word_ids[:, seq_index]
            output_file.write("### Sentence {0}\n".format(num_sentences))
            seq_details = [str(word_id) + ":" + str(logprob)
                for word_id, logprob in zip(seq_word_ids, seq_logprobs)]
            output_file.write(" ".join(seq_details) + "\n")
            output_file.write("Sentence perplexity: {0}\n\n".format(
                numpy.exp(-seq_logprob / seq_length)))

    output_file.write("Number of words: {0}\n".format(num_words))
    output_file.write("Number of sentences: {0}\n".format(num_sentences))
    if num_words > 0:
        cross_entropy = -total_logprob / num_words
        perplexity = numpy.exp(cross_entropy)
        output_file.write("Cross entropy (base e): {0}\n".format(cross_entropy))
        if not log_base is None:
            cross_entropy /= numpy.log(log_base)
            output_file.write("Cross entropy (base {1}): {0}\n".format(
                cross_entropy, log_base))
        output_file.write("Perplexity: {0}\n".format(perplexity))

def _score_utterances(input_file, dictionary, scorer, output_file,
                      log_base=None):
    """Reads utterances from ``input_file``, computes LM scores using
    ``scorer``, and writes one score per line to ``output_file``.

    :type input_file: file object
    :param input_file: a file that contains the input sentences in SRILM n-best
                       format

    :type dictionary: Dictionary
    :param dictionary: dictionary that provides mapping between words and word
                       IDs

    :type scorer: TextScorer
    :param scorer: a text scorer for rescoring the input sentences

    :type output_file: file object
    :param output_file: a file where to write the output n-best list in SRILM
                        format

    :type log_base: int
    :param log_base: if set to other than None, convert log probabilities to
                     this base
    """

    base_conversion = 1 if log_base is None else numpy.log(log_base)

    for line_num, line in enumerate(input_file):
        words = line.split()
        words.append('<sb>')
        
        word_ids = dictionary.words_to_ids(words)
        word_ids = numpy.array([[x] for x in word_ids]).astype('int64')
        
        probs = dictionary.words_to_probs(words)
        probs = numpy.array([[x] for x in probs]).astype(theano.config.floatX)

        lm_score = scorer.score_sentence(word_ids, probs)
        lm_score /= base_conversion
        output_file.write(str(lm_score) + '\n')

        if (line_num + 1) % 100 == 0:
            print("%d sentences rescored." % (line_num + 1))
        sys.stdout.flush()