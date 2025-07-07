from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score


def calculate_metrics(references, hypothesis):

    # BLEU scores
    smoothing_function = SmoothingFunction().method1
    bleu_scores = {
        'BLEU-1': sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing_function),
        'BLEU-2': sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function),
        'BLEU-3': sentence_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function),
        'BLEU-4': sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
    }

    # Rouge-L score
    rouge = Rouge()
    rouge_scores = rouge.get_scores([' '.join(hypothesis)] * len(references), [' '.join(reference) for reference in references], avg=True)['rouge-l']

    # Meteor score
    meteor = meteor_score(references, hypothesis)
    metrics = {
        'BLEU-1': bleu_scores['BLEU-1'],
        'BLEU-2': bleu_scores['BLEU-2'],
        'BLEU-3': bleu_scores['BLEU-3'],
        'BLEU-4': bleu_scores['BLEU-4'],
        'ROUGE-L': rouge_scores['f'],
        'METEOR': meteor
    }

    return metrics

