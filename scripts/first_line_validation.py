"""
Script to validate -- or at least sanity check -- the fine-tuned English first-line model.
"""
import logging
import os
import random
import sys
import time
import json
from typing import List

import numpy as np
import requests
import spacy

import torch
from transformers import (
    MBartTokenizer,
    MBartForConditionalGeneration,
)

from poem_generator.io.candidates import PoemLineList
from quality_estimation.coherence_estimator import SyntacticAnnotator

DEVICE = torch.device('cpu')
BASE_MODEL = "facebook/mbart-large-cc25"
MODEL_FILEPATH = os.path.join(os.path.dirname(__file__), "..", "models/first-line-en-mbart-15000/pytorch_model.bin")
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class DiversityEstimator:
    """Estimates the diversity of a set of poem lines.
    """
    def __init__(self, annotator: SyntacticAnnotator):
        self.annotator = annotator

    def train(self, **kwargs):
        """Dummy implementation.
        """
        pass

    def predict(self, lines: List[str], stopwords: bool = True):
        annotated_lines = [self.annotator.annotate(line) for line in lines]
        if stopwords:
            lemma_lines = [[token.lemma for token in line] for line in annotated_lines]
        else:
            lemma_lines = [[token.lemma for token in line if not token.is_stop] for line in annotated_lines]

        similarities = []
        for i in range(len(lemma_lines)):
            line1 = set(lemma_lines[i])
            for j in range(i+1, len(lemma_lines)):
                line2 = set(lemma_lines[j])
                divisor = len(line1) + len(line2)
                dividend = 2 * len(line1 & line2)
                similarity = dividend / divisor
                similarities.append(similarity)

        average_similarity = sum(similarities) / len(similarities) if len(similarities) > 0 else 1.0
        diversity = 1 - average_similarity
        return diversity


def read_nouns_and_verbs():
    with open(os.path.join(os.path.dirname(__file__), "..", "models", "common-nouns.txt"), 'r') as f:
        lines = f.readlines()
        nouns = [line.strip() for line in lines]

    with open(os.path.join(os.path.dirname(__file__), "..", "models", "common-verbs.txt"), 'r') as f:
        lines = f.readlines()
        verbs = [line.strip() for line in lines]
    return nouns, verbs


def sample_keywords(nouns, verbs, sample_size):
    noun_sample = random.sample(nouns, sample_size)
    verb_sample = random.sample(verbs, sample_size)
    keywords = [" ".join(i for i in e) for e in zip(noun_sample, verb_sample)]
    return keywords


def write_keywords(keywords_list, filepath):
    with open(filepath, 'w') as f:
        for k in keywords_list:
            f.write("{}\n".format(k))


def read_keywords(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        keywords = [line.strip() for line in lines]
    return keywords


def get_words(min_chars=3):
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
    response = requests.get(word_site)
    words = [w.decode() for w in response.content.splitlines()]
    return [w for w in words if len(w) >= min_chars and w != (w[0] * len(w))]


def get_tokenizer():
    tokenizer = MBartTokenizer.from_pretrained(
        BASE_MODEL,
        src_lang="en_XX",
        tgt_lang="en_XX",
    )

    return tokenizer


def get_tokenizer_and_model():
    tokenizer = MBartTokenizer.from_pretrained(
        BASE_MODEL,
        src_lang="en_XX",
        tgt_lang="en_XX",
    )

    logging.info("Loading base model {}".format(BASE_MODEL))
    model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id["en_XX"]

    model.resize_token_embeddings(len(tokenizer))  # is this really necessary here?
    logging.info("Model vocab size is {}".format(model.config.vocab_size))
    model.load_state_dict(torch.load(MODEL_FILEPATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)

    return tokenizer, model


def generate(keywords, tokenizer, model, temperature=1.0) -> PoemLineList:
    """
    Implementation of the first line poem generator using mbart for English language
    :return:
    """
    source = keywords
    encoded = tokenizer.encode(
        source, padding="max_length", max_length=32, truncation=True
    )
    encoded = torch.tensor([encoded] * 10).to(DEVICE)

    sample_outputs = model.generate(
        encoded,
        do_sample=True,
        max_length=16,
        temperature=temperature,
        top_k=5,
    )

    candidates = [
        tokenizer.decode(sample_output, skip_special_tokens=True)
        for sample_output in sample_outputs
    ]
    logging.info("Generated candidates {}".format(candidates))

    return candidates


if __name__ == "__main__":
    sample_size = 5

    #nouns, verbs = read_nouns_and_verbs()
    #keywords = sample_keywords(nouns, verbs, sample_size)
    keywords = read_keywords('keywords_noun_verb_5.txt')
    print("Prepared {} keyword pairs. (first:'{}' last:'{}')".format(len(keywords), keywords[0], keywords[-1]))
    #write_keywords(keywords, 'keywords_noun_verb_{}.txt'.format(sample_size))

    tokenizer, model = get_tokenizer_and_model()
    nlp = spacy.load("en_core_web_sm")
    ann = SyntacticAnnotator(nlp)
    estimator = DiversityEstimator(ann)
    temps = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    generated_lines = {}
    mean_diversities = []
    for temp in temps:
        generated_lines[temp] = []
        all_lines = []
        div_estimates = []
        div_stop_estimates = []
        save_filepath = "first-line-noun-verb-{}-model-15000-temp-{}.txt".format(len(keywords), temp)

        i = 0
        with open(save_filepath, 'w') as f:
            for kw in keywords:
                t = time.monotonic()
                i = i + 1
                line_candidates = generate(kw, tokenizer, model, temperature=temp)
                lines = line_candidates
                generated_lines[temp].append((kw, lines))
                print(lines)
                f.write(str(lines) + "\n")
                div = estimator.predict(lines)
                div_stop = estimator.predict(lines, stopwords=False)
                all_lines.append((kw, lines))
                div_estimates.append(div)
                div_stop_estimates.append(div_stop)
                div_mean = (sum(div_estimates) / len(div_estimates))
                div_stop_mean = (sum(div_stop_estimates) / len(div_stop_estimates))
                t2 = time.monotonic()
                s = "Temp={} {:0>3}: Keywords: '{}' Diversity: {:.4f} ({:.4f}) Running mean: {:.5f} ({:.5f}) ({:.1f}s)"\
                    .format(temp, i, kw, div, div_stop, div_mean, div_stop_mean, t2 - t)
                print(s)
                f.write(s + "\n")

            mean_div = sum(div_estimates) / len(div_estimates)
            mean_div_stop = sum(div_stop_estimates) / len(div_stop_estimates)
            mean_diversities.append((temp, mean_div, mean_div_stop))
            s = "Temp: {} Mean diversity: {} (without stopwords: {})".format(temp, mean_div, mean_div_stop)
            print(s)
            f.write(s + "\n")
            s = "All diversities (std: {}):\n {}".format(np.std(div_estimates), str(div_estimates))
            print(s)
            f.write(s + "\n")
            s = "All diversities without stopwords (std {}):\n {}".format(np.std(div_stop_estimates), str(div_stop_estimates))
            print(s)
            f.write(s + "\n")

    print(mean_diversities)

    with open('lines_noun_verb_{}_model-15000-all.json'.format(len(keywords)), 'w') as f:
        json.dump(generated_lines, f)



