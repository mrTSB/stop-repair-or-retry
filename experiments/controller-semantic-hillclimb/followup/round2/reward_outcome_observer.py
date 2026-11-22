"""Frozen discriminative reward observations; no generation or reference inputs."""
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPOSITORY = 'Skywork/Skywork-Reward-V2-Qwen3-1.7B'
REVISION = 'e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc'
MAX_LENGTH = 1536
SETTINGS = ('score-forest', 'dense-linear-0.01', 'dense-linear-0.1', 'dense-linear-1.0')
PRIOR_WEIGHT = .05


def conversation(row, path, paid):
    if row.get('answer') != '' or path not in ('', 'repair', 'retry') or set(paid) != {'', path}:
        raise ValueError('Only blank-reference paid prefixes may be scored')
    return [{'role': 'user', 'content': row['question']},
            {'role': 'assistant', 'content': paid[path]['text']}]


def tokenize(tokenizer, conversations):
    result = []
    for messages in conversations:
        # The model card calls for no system message and no duplicate BOS.
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        ids = tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
        # Preserve the problem and response ending if a paid observation is long.
        kept = ids if len(ids) <= MAX_LENGTH else ids[:MAX_LENGTH//2] + ids[-MAX_LENGTH//2:]
        result.append({'input_ids': kept, 'original_tokens': len(ids), 'truncated': len(ids) > MAX_LENGTH})
    return result


def load(directory, device):
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='sdpa').eval().to(device)
    if model.config.model_type != 'qwen3' or model.config.num_labels != 1 or model.config.hidden_size != 2048:
        raise ValueError('Changed frozen reward architecture')
    model.requires_grad_(False)
    return model, tokenizer


def score_batch(model, tokenizer, entries, device):
    batch = tokenizer.pad([{'input_ids': r['input_ids']} for r in entries], padding=True, return_tensors='pt')
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        result = model(**batch, output_hidden_states=True, use_cache=False)
        # Match Qwen3 sequence classification's last non-padding token pooling.
        index = (batch['input_ids'] != model.config.pad_token_id).to(torch.int64)
        index = (torch.arange(index.shape[1], device=device) * index).argmax(-1)
        hidden = result.hidden_states[-1][torch.arange(len(entries), device=device), index]
        projected = model.score(hidden).reshape(-1)
        torch.testing.assert_close(projected, result.logits.reshape(-1), rtol=1e-4, atol=1e-4)
    return result.logits.float().cpu().numpy().reshape(-1), hidden.float().cpu().numpy()


@dataclass
class Forecaster:
    settings: str
    scaler: object
    pca: object
    estimator: object
    prior: np.ndarray

    def matrix(self, base, scores, hidden):
        x = np.column_stack((base, scores))
        if self.pca is None:
            return x
        return np.column_stack((self.scaler.transform(x), self.pca.transform(hidden)))

    def predict(self, base, scores, hidden, valid):
        probability = np.tile(PRIOR_WEIGHT*self.prior, (len(base), 1))
        probability[:, self.estimator.classes_.astype(int)] += (1-PRIOR_WEIGHT)*self.estimator.predict_proba(self.matrix(base, scores, hidden))
        probability[~np.asarray(valid, bool), 8:] = 0.
        return probability/probability.sum(1, keepdims=True)


def fit(base, scores, hidden, labels, settings):
    if settings not in SETTINGS or len(set(map(len, (base, scores, hidden, labels)))) != 1:
        raise ValueError('Aligned registered feature recipe required')
    counts = np.bincount(labels, minlength=16)+1.
    model = Forecaster(settings, None, None, None, counts/counts.sum())
    if settings == 'score-forest':
        model.estimator = ExtraTreesClassifier(n_estimators=96, max_depth=8, min_samples_leaf=24,
            max_features=.7, n_jobs=1, random_state=2381131)
    else:
        model.scaler = StandardScaler().fit(np.column_stack((base, scores)))
        model.pca = PCA(n_components=32, whiten=True, svd_solver='randomized', random_state=2381131).fit(hidden)
        model.estimator = LogisticRegression(C=float(settings.rsplit('-', 1)[1]), max_iter=2000)
    model.estimator.fit(model.matrix(base, scores, hidden), labels)
    return model
