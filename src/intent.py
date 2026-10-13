"""
Intent matching on top of the ASR transcription (commands dataset, SLURP intents).

A pretrained text classifier (cartesinus/slurp-intent_baseline-xlm_r-en, XLM-R base fine-tuned on SLURP, 93 intents,
no training in this repo) maps a transcription to a SLURP intent (scenario_action, e.g. alarm_query).
The ASR model followed by this classifier is the system under test of the intent matching task: its output
(intent probabilities of the transcription) is a valid black-box signal for the attacks (INTENT_TARGET objective).
"""
import numpy as np
import torch

INTENT_MODEL = "cartesinus/slurp-intent_baseline-xlm_r-en"
_models = {}


def _load(device: str):
    if device not in _models:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL).to(device).eval()
        _models[device] = (tokenizer, model)
    return _models[device]


def intent_labels(device: str = "cpu") -> list:
    """The 93 intents of the classifier, in the column order of intent_probabilities."""
    id2label = _load(device)[1].config.id2label
    return [id2label[i] for i in range(len(id2label))]


@torch.no_grad()
def intent_logits(texts: list, device: str = "cpu", batch_size: int = 32) -> np.ndarray:
    """Raw classifier logits over the 93 intents for every text: array [len(texts), 93]."""
    tokenizer, model = _load(device)
    logits = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer([t or "" for t in texts[start:start + batch_size]], padding=True, truncation=True,
                          return_tensors="pt").to(device)
        logits.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(logits) if logits else np.zeros((0, len(intent_labels(device))))


@torch.no_grad()
def intent_probabilities(texts: list, device: str = "cpu", batch_size: int = 32) -> np.ndarray:
    """Softmax over the 93 intents for every text: array [len(texts), 93]."""
    tokenizer, model = _load(device)
    probabilities = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer([t or "" for t in texts[start:start + batch_size]], padding=True, truncation=True,
                          return_tensors="pt").to(device)
        probabilities.append(torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.zeros((0, len(intent_labels(device))))


def classify_intents(texts: list, device: str = "cpu") -> list:
    """Predicted (most likely) intent for every text."""
    labels = intent_labels(device)
    return [labels[i] for i in intent_probabilities(texts, device).argmax(-1)]


def target_probability(texts: list, target_intent: str, device: str = "cpu") -> list:
    """Probability of the target intent for every text (1 = the SUT outputs the target intent)."""
    column = intent_labels(device).index(target_intent)
    return intent_probabilities(texts, device)[:, column].tolist()


def target_margin(texts: list, target_intent: str, device: str = "cpu") -> list:
    """Logit of the target intent minus the largest other logit, per text: > 0 exactly when the target is the predicted intent."""
    logits = intent_logits(texts, device)
    column = intent_labels(device).index(target_intent)
    target = logits[:, column].copy()
    logits[:, column] = -np.inf
    return (target - logits.max(-1)).tolist()
