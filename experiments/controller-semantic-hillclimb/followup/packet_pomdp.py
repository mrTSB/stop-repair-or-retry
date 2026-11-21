"""Eight-state POMDP retaining one ordinary brief review and its worked solution."""

import json
from pathlib import Path

import numpy as np

from mentor_pomdp import fit as original_fit, evaluate
from root_medium_pomdp import training_observations
from packet_bank import view
from difficulty_pomdp import FIXED
from scoring import correct


def fit(*args, **kwargs):
    model = original_fit(*args, **kwargs)
    rows, trees, judges, directory = args[:4]
    counts = np.zeros(8)
    prices = []
    recipe = json.loads((Path(directory) / 'recipe.json').read_bytes())
    encoder = recipe['parameters'] / 1e9 * recipe['max_length'] / 3.09 if kwargs.get('semantic_review', False) else 0.
    for row in rows:
        qid = row['id']
        if (qid, '') not in judges:
            continue
        hard = int(np.mean([correct(trees[qid][p], row['answer']) for p in FIXED]) < .6)
        judge = judges[qid, '']
        s = 4 * hard + 2 * int(correct(trees[qid][''], row['answer'])) + int(correct(judge, row['answer']))
        counts[s] += 1
        prices.append(3 * (judge['input_tokens'] + judge['output_tokens']) / 3.09 + encoder)
    # The inherited deeper-review model uses 2800 for wholly unobserved states.
    # This protocol buys an ordinary brief review, whose actual pooled price is
    # the appropriate fallback. States with observations retain the fitted price.
    for dynamics in model.dynamics.values():
        dynamics.review_costs[counts == 0] = float(np.mean(prices))
    model.fit_receipt.update(root_full_packet=True,
                             root_review='Exact ordinary brief root review; no deeper LLM reasoning',
                             feedback='Retain the full brief worked solution, independent answer, and check',
                             guided_format='Final first, at most35justification words,128output tokens',
                             maximum_logical_reviews=1,
                             unseen_review_cost='Pooled actual brief-review price, including its encoder pass')
    return model
