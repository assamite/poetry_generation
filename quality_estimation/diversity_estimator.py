from typing import List

from quality_estimation.coherence_estimator import SyntacticAnnotator


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
            lemmatised_lines = [[token.lemma for token in line] for line in annotated_lines]
        else:
            lemmatised_lines = [[token.lemma for token in line if not token.is_stop] for line in annotated_lines]

        print(lemmatised_lines)
        similarities = []
        for i in range(len(lemmatised_lines)):
            line1 = set(lemmatised_lines[i])
            for j in range(i+1, len(lemmatised_lines)):
                line2 = set(lemmatised_lines[j])
                union = line1 | line2
                intersection = line1 & line2
                similarity = len(intersection) / len(union)
                similarities.append(similarity)

        average_similarity = sum(similarities) / len(similarities)
        diversity = 1 - average_similarity
        return diversity


if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm")
    ann = SyntacticAnnotator(nlp)
    estimator = DiversityEstimator(ann)

    lines = ["This is a test.", "This was not a test.", "This will be a test."]
    div = estimator.predict(lines)
    print(div)



    
