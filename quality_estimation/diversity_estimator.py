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

        average_similarity = sum(similarities) / len(similarities)
        diversity = 1 - average_similarity
        return diversity


if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm")
    ann = SyntacticAnnotator(nlp)
    estimator = DiversityEstimator(ann)

    lines = ["This is a test.", "This should not be a test.", "This will be a test."]
    div = estimator.predict(lines)
    print(div)

    lines = ["This is a test.", "This is a test.", "This will be a test."]
    div = estimator.predict(lines)
    print(div)

    lines = ["This is a test.", "This is a test.", "This is a test."]
    div = estimator.predict(lines)
    print(div)



    
