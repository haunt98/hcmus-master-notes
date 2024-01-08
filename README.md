# HCMUS Master AI

## Learning problems

### Supervised learning

### Unsupervised learning

### Reinforcement learning

Agent learn to interact environment based on **feedback** signals.

Steps:

- Define goal and reward system.
- Trial and error by taking action then observing results.
- Learning by adapting behavior based on collected rewards.
- Reaching goal by maximizing rewards.

Keys:

- No instructions, self learn by interaction and feedback.

Good at:

- Games like chess, go.

**Monte Carlo tree search**: combine tree search and random sampling.

## Learning techniques

### Multi-task learning

Instead of buiding separate models for each task, Multi-task learning use single
model with shared layers.

### Active learning

### Online learning

Model is updated as new data arrives rather then waiting for final decision (the
end, which can never happen)

### Transfer learning

Model learning from one problem then applied to **different but related**
problem.

Use for:

- Classifier
- Feature extractor
- Weight initialization

### Ensemble learning

Combine multi predictions from multi models aka base learners then output final
prediction.

Types:

- Bagging: majority votes over minority.
- Boosting: weighting votes.
- Stacking: meta-models to learn from votes.

Random Forest

AdaBoost

Use for:

- Deepfake detection

## Advance learning

### Federated learning

### Explainable AI (XAI)
