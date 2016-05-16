#!/usr/bin/python
# -*- coding: utf-8 -*-

import gensim
import logging
import numpy as np

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # load vanilla model
    vanilla_model = "./models/vanilla/wikipedia.vanilla.w2v"
    model = gensim.models.Word2Vec.load(vanilla_model)

    model.init_sims(replace=True)

    # algebric operations with words
    print(model.most_similar(positive=["mulher", "rei"],
                             negative=["homem"]))
    # most similar using multiplicative combination
    print(model.most_similar_cosmul(positive=["mulher", "rei"],
                                    negative=["homem"]))
    # out of context word
    print(model.doesnt_match("Portugal Espanha Alemanha Pacífico".split()))

    # cosine similarity
    print(model.similarity("homem", "mulher"))

    # cosine similarity between two set of words
    print(model.n_similarity(["Portugal", "português"],
                             ["lisboa", "bacalhau"]))

    # print word vector
    print(model["palavra"])

    # model size (vocabulary, vector size)
    print(model.syn0.shape)

    # convert vocabulary to  aset
    vocab = set(model.index2word)

    # word in set?
    print("vetusto" in vocab)
    print("australopitecos" in vocab)
    print("vocabulário" in model.vocab)

    # find word from vector
    def find_nearest(model, vector, K=5):
        square = np.square(model.syn0norm - vector)
        idx = np.sum(square, axis=1).argsort()[:K]
        return map(lambda x: model.index2word[x], idx)

    print(find_nearest(model, model["palavra"]))
    # or
    print(model.most_similar(positive=[model["palavra"]], topn=5))

    # find antonyms
    print(model.most_similar(positive=['bom', 'triste'],
                             negative=['mau']))

    # export to word2vec format
    # model.save_word2vec_format("word2vec.format")
