from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

def rouge_score(hypotheses: str, references: str):
    return rouge_scorer.score(references, hypotheses)

def bleu_score(hypotheses: str, references: str):
    return sentence_bleu(references, hypotheses)