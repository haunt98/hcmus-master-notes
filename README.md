# HCMUS Master AI

## Decision Tree

Decision Trees are often used to answer that kind of question: given a labelled
dataset, how should we classify new samples?

- Labelled: Our dataset is labelled because each point has a class (color): blue
  or green.
- Classify: To classify a new datapoint is to assign a class (color) to it.

How to train/build a decision tree: Determine the root node ...

Intuitively, we want a decision node that makes a “good” split, where “good” can
be loosely defined as separating different classes as much as possible.

Entropy can be roughly thought of as how much variance the data has. For
example:

- A dataset of only blues would have very low (in fact, zero) entropy.
- A dataset of mixed blues, greens, and reds would have relatively high entropy.

**Information Gain** = how much Entropy we removed

Information Gain is calculated for a split by subtracting the weighted entropies
of each branch from the original entropy. When training a Decision Tree using
these metrics, the best split is chosen by **maximizing** Information Gain.

Information Gain Ratio: to avoid bias toward multi-branch splits.

**Pruning tree**: remove branches.

Why?

- Less complex, easier to understand.
- Classify faster.

How?

- Pre-pruning: stop growing tree early.
- Post-pruning: grow full tree then remove branches.

### References

- [A Simple Explanation of Information Gain and Entropy](https://victorzhou.com/blog/information-gain/)
- [A Simple Explanation of Gini Impurity](https://victorzhou.com/blog/gini-impurity/)
- [Random Forests for Complete Beginners](https://victorzhou.com/blog/intro-to-random-forests/)

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

Perceptron Learning Algorithm (PLA) (Luật huấn luyện perceptron):

- Init randomly weight
- If wrong re-update weight: `wi = wi + delta wi`

**Gradient Descent**

- Start at random point
- Calculate the slope (gradient) at that point
- Move in the direction of the negative gradient
- Stop when reach local minimum

**Delta Rule**: based on gradient descent

- Calculate error
- Use error to calculate delta for each weight
- Update weight, moving closer to desired output

Loss function (Hàm lỗi) (Hàm mất mát):

### References

- [What is Perceptron: A Beginners Guide for Perceptron](https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron)
- [Bài 9: Perceptron Learning Algorithm](https://machinelearningcoban.com/2017/01/21/perceptron/)
- [Bài 7: Gradient Descent (phần 1/2)](https://machinelearningcoban.com/2017/01/12/gradientdescent/)

## Multi Layer Perceptron (MLP)

Single perceptron can only handle linear, but MLP can handle non-linear easily.

**Sigmoid function**: convert input to output in [0, 1]

- If input is large, output is close to 1
- If input is small, output is close to 0

**Back Propagation**

- Init weight with small value
- At each layer
  - Calculate weighted sum of inputs
  - Apply activation function
- Continue until reach output layer
- Backward Pass (Backpropagation)
  - Calculate using error function
  - Update weight

### References

- [Bài 14: Multi-layer Perceptron và Backpropagation](https://machinelearningcoban.com/2017/02/24/mlp/)

## Adam Algorithm

Gradient Descent with Momentum: like physics, to bypass unwanted local minimum
to reach another local minimum.

Nesterov Accelerated Gradient (NAG): improve Momentum.

Why? Momentum, when go near local minimum, will slow down for a while, danging
near local minimum. NAG will help Momentum to converge faster to reach local
minimum.

The idea is look ahead 1 step. Instead of using current position, use next
position to calculate gradient.

Adaptive Learning Rate: learning rate is updated during training.

AdaGrad (Adaptive Gradient Algorithm)

- Adapt learning rate by scaling them inversely proportional to the sum of the
  historical squared values of the gradient
- Sum of history squared values will be big then learning rate will be small.

RMSProp: improve AdaGrad

- Change gradient accumulation into exponentially weighted moving average (use
  decay rate)
- Converges rapidly when applied to convex function

Adam (Adaptive Moment): combine RMSProp with Momentum

### References

- [Bài 8: Gradient Descent (phần 2/2)](https://machinelearningcoban.com/2017/01/16/gradientdescent2/)

## Learning problems

### Supervised learning

### Unsupervised learning

### Reinforcement learning

Agent learn to interact environment based on **feedback** signals.

Steps:

- Define goal and reward system.
- Trial and error by taking action then observing results.
  - **Partial observation** of state of the world.
  - Then pick one of possible actions -> let environment change from one state
    to another.
- Learning by adapting behavior based on collected rewards.
- Reaching goal by maximizing rewards.

Concepts:

- State `s` is a **complete** description of state of the world.
- Observation `o` is a **partial** description of a state.
  - Fully or partially observable environment: like chess or poker.
- Action space: all valid actions in given environment.
  - Discrete action space: finite number of moves are available like chess
    moves.
  - Continuous action space: defined by real-valued parameters like robot arm
    movement.
  - Hybrid: combine discrete and continuous.
- Policy: map given state -> action to be taken.
  - Can be simple function, dictionary or search process.
  - Deterministic policy: always same output action given same input state.
  - Stochastic policy: output a probability distribution over the possible
    actions.
  - Finding the optimal policy is the core challenge of reinforcement learning.
- Trajectory is a sequence status and actions (s0, a1, s1, a1, ...)
- Rewards usually numeric values sent from the environment.
  - Reward signal: reward function (st, st, s(t+1)), maybe stochastic function.
  - Finite-horizon undiscounted return: sum rewards in fixed window of steps.
  - Infinite-horizon discounted return: sum rewards but reward in the future get
    discount depends on how far off in the future.
- Expected return:
  - Consider choice of policy and any reward measure.
  - Select a policy which maximizes expected return.
- Value of a state is total amout reward an agents **hope** to accumulate in the
  future.
  - Value function may find most suitable solution for specific contexts that
    have trade-off.
  - We care about value than reward, of course determine value is harder than
    reward.
- Model of environment: how environment will behave.
  - Model-free methods: trial-and-error learning.
  - Model-based methods: use models and planning.

Keys:

- No instructions, self learn by interaction and feedback.

Good at:

- Games like chess, go.

**Monte Carlo tree search**: combine tree search and random sampling.

**Markove Deision Processes (MDP)**

- Framework for planning and maximizing your future gains in scenarios where
  some things are out of your control but you still have choices to make.
- The future state only depends on the current state and the action taken, not
  on the history of previous states and actions

**Model-Based Learning**

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

## ?? Học tổ hợp

Condorcet's Jury

Majority Voting

- Weak leaner
- Comnine weak learner to strong learner
- Learn bias and variance

Occam's Razor

Adaboost

### References

- [AdaBoost: A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*](https://www.sciencedirect.com/science/article/pii/S002200009791504X)
