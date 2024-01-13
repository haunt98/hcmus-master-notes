# HCMUS Master AI

## Perceptron

Basic building block of neuron networks. Use for supervised learning of binary
classifiers.

Steps:

- Input `xi` with weight `wi`
- Sum: `w0 + w1*x1 + ... + wi*xi`
- Activation function
- Output

Can represent basic bool: AND, OR, NAND, NOR,... but can not represent XOR
because can not divide XOR by a line which cross (0, 0) aka not linearly
separable.

Delta Learning Rule:

- Init randomly
- Calculate ...
- Compare with desired output
- If wrong re-update weight: `wi = wi + delta wi`

Gradient descent:

Use error function. ???

See:

- [What is Perceptron: A Beginners Guide for Perceptron](https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron)

### Multi Layer Perceptron (MLP)

Single perceptron can only handle linear, but MLP can handle non-linear easily.

Controlled parameters. ???

Full connected ???

Sigmoid function ???

Back propagation:

- Init weight with small value
- Calculate ???

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

## Programming

### References

- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
