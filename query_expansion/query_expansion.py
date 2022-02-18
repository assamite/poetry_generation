import logging
from itertools import permutations

from gensim import models


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

model_path = "../models/GoogleNews-vectors-negative300.bin"
model = models.KeyedVectors.load_word2vec_format(model_path, binary=True)


def sanitize_keywords(keywords):
    return keywords.lower().replace("_", " ")


def to_word2vec_token(token):
    return token.replace(" ", "_")


def clean_word2vec_tokens(token_list):
    return [clean_word2vec_token(t) for t in token_list]


def clean_word2vec_token(token):
    return token.replace("_", " ")


def permute_words(words):
    """Get all permutations of the given word list (['a', 'b', 'c', ...]).

    Current implementation is brute force and does not take into account repetitions in the list.
    """
    return list(permutations(words))


def word_lists_to_keywords(words):
    """Convert a list of word lists ([('a', 'b'), ('c', 'd', 'e'), ...]) into a list of keywords.
    """
    return list(map(lambda x: " ".join(w for w in x), words))


def combine_queries(query_dict):
    #query_dict['all'] = [query_dict['original']]
    query_dict['all'] = word_lists_to_keywords(query_dict['permutations'])
    print(query_dict['all'])


def expand(keywords, model, permute=True, similar=True, negative=True):
    queries = {'original': keywords}
    keywords = sanitize_keywords(keywords)
    queries['sanitized'] = keywords
    words = keywords.split()
    if permute:
        queries['permutations'] = permute_words(words)
    if similar:
        queries['similar'] = {}
        for word in words:
            similar = get_similar_words(word, model)
            if similar is not None:
                print(similar)
                queries['similar'][word] = clean_word2vec_tokens(similar)

    if negative:
        # TODO: generalise to multiple words
        if len(words) == 2:
            queries['negative'] = {}
            similar_with_negative = get_similar_words(words[0], model, neg_word=words[1])
            queries['negative'][words[0]] = clean_word2vec_tokens(similar_with_negative)
            similar_with_negative = get_similar_words(words[1], model, neg_word=words[0])
            queries['negative'][words[1]] = clean_word2vec_tokens(similar_with_negative)
    return queries


def get_similar_words(pos_word, model, neg_word=None, topn=5):
    similar = None
    try:
        if neg_word is not None:
            similar = model.most_similar(pos_word, neg_word, topn=topn)
        else:
            similar = model.most_similar(pos_word, topn=topn)
    except KeyError:
        if neg_word is None:
            logging.info("Word2Vec model did not have '{}' in its vocabulary, not expanding the keyword."
                         .format(pos_word))
        else:
            logging.info("Word2Vec model did not have '{}' or '{}' in its vocabulary, not expanding the keyword."
                         .format(pos_word, neg_word))

    return [w[0] for w in similar]


if __name__ == "__main__":
    keywords = "winning run"
    queries = expand(keywords, model)
    print(expand(keywords, model))
    combine_queries(queries)


