# Question-level cross-validation: calibration and context

This separate experiment tests whether safer handling of unsupported observation categories improves the two-state POMDP. It does not add answer-change observations, alter hidden-state count, evaluate test, or spend inference credits. The scope question about adding answer-change information was still pending during this work.

## Protocol

Five deterministic folds split the 100 training questions by hashed question ID, balanced within domain. Each fold fits 80 questions and evaluates 20; all seven branches of a question remain together. Training base IDs and normalized question texts are unique. All fold priors, transitions, observations, context dynamics and costs are re-estimated from the fitting 80, not copied from the 100-question model.

Three declared variants were evaluated at the two existing penalties 0.01 and0.248: pooled dynamics; context-conditioned dynamics with shrinkage 10; and the same context model with unsupported-category calibration. The selected variant maximizes out-of-fold accuracy, then minimizes tokens, then uses a fixed variant order and smaller penalty. Selection is completed before loading dev. Only the selected variant is evaluated on dev. This is not a sweep over new penalties or shrinkage values.

## Calibration change

For an observation symbol absent from all fitting examples at a root/action distribution, both hidden states receive the same positive emission probability. Thus seeing that symbol preserves the prior/predicted belief rather than creating unsupported evidence for correctness. Supported symbols are renormalized. Context emissions apply this safeguard to symbols absent from the pooled action samples, while retaining the declared pooled shrinkage for locally sparse categories.

This corrects an inference behavior; it does not add information or imply a particular correctness probability for low confidence. Four analytic tests verify normalization, no-information updates, and fold isolation.

## Out-of-fold training results

These are 100 held-out predictions per configuration, each from a model that excluded that question. They are not training-set resubstitution scores.

| Variant | Lambda | Correct | Mean tokens |
|---|---:|---:|---:|
| pooled | 0.01 | 57/100 | 1316.53 |
| pooled | 0.248 | 50/100 | 777.69 |
| context | 0.01 | 57/100 | 1302.75 |
| context | 0.248 | 50/100 | 750.04 |
| context_calibrated | 0.01 | 57/100 | 1304.77 |
| context_calibrated | 0.248 | 50/100 | 750.04 |

Fixed repair twice gets 57/100 on these same training questions; fixed repair once gets 54/100. None of the fitted configurations establishes an accuracy advantage over the corresponding strong fixed baseline. Cross-validation estimates have sampling variability and are not independent across model variants.

## Selection and development result

The declared tie-break selects **context dynamics at lambda 0.01**: 57/100 correct with 1302.75 mean tokens. Fitting that selected variant on all 100 training questions and evaluating all 60 dev questions gives **32/60 correct (53.33%) and 1253.03 mean tokens**. It repairs twice on all 60, so it again exactly matches repair twice. The calibrated variant was not selected, and it was not separately tried on dev after seeing this outcome.

The earlier context model at lambda 0.248 had a small dev advantage over the pooled model. This cross-validation result shows that choice is not strongly supported across training folds: the 0.248 variants score 50/100 out of fold, below fixed repair once at 54/100. This does not prove any model is universally worse; it does warn against treating a small dev gain as robust progress.

## What this means for improvement

Calibration alone did not improve measured accuracy. The unsupported-category rule has a sound no-information interpretation, but better mechanics are not the same as better predictions. The model still receives only coarse confidence/validity observations, and most roots share the same high-confidence observation.

The next substantive experiment would retain two hidden states and add history-aware answer-change observations, if the user selects that scope. Because agreement depends on previous and current answers, its model must condition on the transition/history rather than naively treat it as independent evidence of next-state correctness. Use a new protocol and question-level training validation, preserve the confidence-only comparisons, and evaluate a single selected candidate on dev. Four states and new test evaluation are not part of this experiment.

## Evidence and limitations

Protocol, folds, all 15 fitted fold models, all 30 held-out path sets, OOF summaries, selected full-training model and dev paths are saved. The independent verifier reconstructs fitting, fold isolation, prediction paths, selection and results. Selection and past diagnostic directions have been influenced by previously inspected dev/test results, so neither these dev outcomes nor a future reuse of that test set constitutes an untouched confirmation claim. No paid calls were made, and all earlier frozen artifacts are preserved.
