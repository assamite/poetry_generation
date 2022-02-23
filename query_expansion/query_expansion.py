from copy import copy
import logging
from itertools import permutations

from gensim import models


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

model_path = "../models/GoogleNews-vectors-negative300.bin"
model = models.KeyedVectors.load_word2vec_format(model_path, binary=True)


def sanitize_words(word_list):
    return [word.lower().replace("_", " ") for word in word_list]


def to_word2vec_token(token):
    return token.replace(" ", "_")


def clean_word2vec_tokens(token_list):
    return [clean_word2vec_token(t) for t in token_list]


def clean_word2vec_token(token):
    return token.lower().replace("_", " ")


def permute_words(words):
    """Get all permutations of the given word list (['a', 'b', 'c', ...]).

    Current implementation is brute force and does not take into account repetitions in the list.
    """
    return list(permutations(words))


def word_lists_to_keywords(word_lists):
    """Convert a list of word lists ([('a', 'b'), ('c', 'd', 'e'), ...]) into a list of keywords (['a b', 'c d e', ...]).
    """
    return list(map(lambda x: " ".join(w for w in x), word_lists))


def replace_all_similar(word_list, similar_dict):
    similar_queries = {}
    for word, similar_words in similar_dict.items():
        similar_queries[word] = replace_with_similar(word_list, word, similar_words)
    return similar_queries


def replace_with_similar(word_list, word, similar_words):
    new_word_lists = []
    try:
        word_index = word_list.index(word)
    except ValueError:
        raise ValueError("'{}' not in word list: {}".format(word, word_list))

    for w in similar_words:
        if w != word:
            new_list = copy(word_list)
            new_list[word_index] = w
            new_word_lists.append(new_list)
    return new_word_lists


def combine_queries(query_dict):
    if 'permutations' in query_dict:
        query_dict['all'] = word_lists_to_keywords(query_dict['permutations'])
        print(query_dict['all'])
    else:
        keywords = " ".join(w for w in query_dict['original'])
        query_dict['all'] = [keywords,]

    if 'similar_words' in query_dict:
        similar_queries = replace_all_similar(query_dict['sanitized'], query_dict['similar_words'])
        print(similar_queries)
        for word, word_lists in similar_queries.items():
            keywords_list = word_lists_to_keywords(word_lists)
            query_dict['all'].extend(keywords_list)
        print(len(query_dict['all']), query_dict['all'])

    if 'similar_words_negative' in query_dict:
        similar_queries = replace_all_similar(query_dict['sanitized'], query_dict['similar_words_negative'])
        print(similar_queries)
        for word, word_lists in similar_queries.items():
            keywords_list = word_lists_to_keywords(word_lists)
            query_dict['all'].extend(keywords_list)
        print(len(query_dict['all']), query_dict['all'])

    query_dict['all'] = list(set(query_dict['all']))
    print(len(query_dict['all']), query_dict['all'])
    return query_dict


def expand(keywords, model, permute=False, similar=True, negative=True):
    words = keywords.split()
    queries = {'original': words}
    words = sanitize_words(words)
    queries['sanitized'] = words
    words = keywords.split()
    if permute:
        queries['permutations'] = permute_words(words)
    if similar:
        queries['similar_words'] = {}
        for word in words:
            similar = get_similar_words(word, model)
            if similar is not None:
                queries['similar_words'][word] = clean_word2vec_tokens(similar)

    if negative:
        # TODO: generalise to multiple words
        if len(words) == 2:
            queries['similar_words_negative'] = {}
            similar_with_negative = get_similar_words(words[0], model, neg_word=words[1])
            queries['similar_words_negative'][words[0]] = clean_word2vec_tokens(similar_with_negative)
            similar_with_negative = get_similar_words(words[1], model, neg_word=words[0])
            queries['similar_words_negative'][words[1]] = clean_word2vec_tokens(similar_with_negative)
    return queries


def _get_similar_non_capitalized_words(pos_word, model, neg_word=None, topn=5):
    words = []
    i = 0
    max_i = topn * 2
    while len(words) < topn:
        similar = model.most_similar(pos_word, neg_word, topn=max_i)
        logging.info("Retrieved {} similar words for {} (neg:{}), checking from {} to {}"
                     .format(max_i, pos_word, neg_word, i, max_i))
        for j in range(i, max_i):
            word_candidate = similar[j][0]
            if word_candidate not in words and not any(x.isupper() for x in word_candidate):
                words.append(word_candidate)
                if len(words) == topn:
                    break
        i = max_i
        max_i = max_i * 2
    print(words)
    return words


def get_similar_words(pos_word, model, neg_word=None, topn=5, filter_capitalized=True):
    """Get words similar to the given word from the Word2Vec model.

    :param str pos_word: "positive example" word
    :param model: Gensim Word2Vec model, or similar model with :meth:`most_similar`
    :param str neg_word: "negative example" word
    :param int topn: How many similar words to return
    :param bool filter_capitalized: True if the function should not return capitalized words, False otherwise
    """
    similar = None
    try:
        if filter_capitalized:
            similar = _get_similar_non_capitalized_words(pos_word, model, neg_word, topn)
        else:
            similar = [w[0] for w in model.most_similar(pos_word, neg_word, topn=topn)]
    except KeyError:
        if neg_word is None:
            logging.info("Word2Vec model did not have '{}' in its vocabulary, not expanding the keyword."
                         .format(pos_word))
        else:
            logging.info("Word2Vec model did not have '{}' or '{}' in its vocabulary, not expanding the keyword."
                         .format(pos_word, neg_word))

    return similar


if __name__ == "__main__":
    keywords = "distressed owl"
    queries = expand(keywords, model)
    print(expand(keywords, model))
    combine_queries(queries)


