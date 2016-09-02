#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import gensim
import argparse


if __name__ == "__main__":
    """
    Evaluates a given word embedding model.

    To use:
    evaluate.py path_to_model [-restrict]
    optional restrict argument performs an evaluation using the original
    Mikolov restriction of vocabulary
    """

    desc = "Evaluates a word embedding model"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-m",
                        required=True,
                        help="model")
    parser.add_argument("-t",
                        nargs="?",
                        default="./testset/LX-4WAnalogies.txt",
                        help="testset")
    parser.add_argument("-r",
                        nargs="?",
                        default=False,
                        help="Vocabulary restriction")
    args = parser.parse_args()

    model_path = args.m
    testset = args.t

    # use restriction?
    restriction = None
    if args.r:
        restriction = 30000

    # set logging definitions
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # load and evaluate
    model = gensim.models.Word2Vec.load(model_path)
    model.accuracy(testset, restrict_vocab=restriction)
