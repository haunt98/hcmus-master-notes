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
- [Decision Trees Explained — Entropy, Information Gain, Gini Index, CCP Pruning](https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c)

## Perceptron

Basic building block of neuron networks. Use for supervised learning of binary
classifiers.

Can represent basic bool: AND, OR, NAND, NOR,... but can not represent XOR
because can not divide XOR by a line which cross (0, 0) aka not linearly
separable.

Perceptron Learning Algorithm (PLA) (Luật huấn luyện perceptron):

- Input `xi` with random weight `wi`
- Sum: `w0 + w1*x1 + ... + wi*xi`
- Activation function
- Calculate delta
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

Loss function (Hàm lỗi, Hàm mất mát):

- Mean Square Error (MSE)

### References

- [Bài 9: Perceptron Learning Algorithm](https://machinelearningcoban.com/2017/01/21/perceptron/)
- [Bài 7: Gradient Descent (phần 1/2)](https://machinelearningcoban.com/2017/01/12/gradientdescent/)
- [Optimization: Stochastic Gradient Descent](https://cs231n.github.io/optimization-1/)
- [What is Perceptron: A Beginners Guide for Perceptron](https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron)

## Multi Layer Perceptron (MLP)

Single perceptron can only handle linear, but MLP can handle non-linear easily.

**Sigmoid function**: convert input to output in [0, 1]

- If input is large, output is close to 1
- If input is small, output is close to 0

Derivative of sigmoid function: `f'(x) = f(x) * (1 - f(x))`

**Tanh (tangent hyperbolic) function**: convert input to output in [-1, 1]

- If input is large, output is close to 1
- If input is small, output is close to -1

**Back Propagation**

- Init weight with small, random value
- At each layer
  - Calculate weighted sum of inputs
  - Apply activation function
- Continue until reach output layer
- Backward Pass (Backpropagation)
  - Calculate using error function
  - Update weight

Quirk:

- Different init weight can lead to different result

### References

- [Bài 14: Multi-layer Perceptron và Backpropagation](https://machinelearningcoban.com/2017/02/24/mlp/)
- [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
- [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
- [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)
- [CS231n Winter 2016: Lecture 5: Neural Networks Part 2](https://www.youtube.com/watch?v=gYpoJMlgyXA)
- [Activation Functions: Sigmoid vs Tanh](https://www.baeldung.com/cs/sigmoid-vs-tanh-functions)

## Gradient Descent Optimizer

> Gradient indicate the direction of the steepest increase.

Gradient Descent with Momentum: like physics, to bypass unwanted local minimum
to reach another local minimum.

Nesterov Accelerated Gradient (NAG): improve Momentum.

Why? Momentum, when go near local minimum, will slow down for a while, danging
near local minimum. NAG will help Momentum to converge faster to reach local
minimum.

The idea is look ahead 1 step. Instead of using current position, use next
position to calculate gradient.

Adaptive Learning Rate: learning rate is updated during training.

- High learning rate: move quickly, may overshoot the minimum, unstable converge
- Low learning rate: move slowly, may stuck in local minimum, slow converge

AdaGrad (Adaptive Gradient Algorithm)

- Adapt learning rate by scaling them inversely proportional to the sum of the
  historical squared values of the gradient
- Sum of history squared values will be big then learning rate will be small.

RMSProp: improve AdaGrad by adding decay factor.

- Change gradient accumulation into exponentially weighted moving average (use
  decay rate)
- Converges rapidly when applied to convex function

Adam (Adaptive Moment): get ability to adapt gradident from RMSProp and speed
from Momentum.

> Moreover, in areas where the gradient (the slope of the loss function) changes
> rapidly or unpredictably, Adam takes smaller, more cautious steps. This helps
> avoid overshooting the minimum. Instead, in areas where the gradient changes
> slowly or predictably, it takes larger steps. This adaptability is key to
> Adam’s efficiency, as it navigates the loss landscape more intelligently than
> algorithms with a fixed step size.

### References

- [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)
- [Bài 8: Gradient Descent (phần 2/2)](https://machinelearningcoban.com/2017/01/16/gradientdescent2/)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [The Math Behind the Adam Optimizer](https://towardsdatascience.com/the-math-behind-adam-optimizer-c41407efe59b)
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs)
- [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)

## Bagging and Boosting (Ensemble Learning)

> Boosting is the problem of using the notion of weak learning and boost the
> performance of weak learners to a strong learne

- Bias: how wrong the expected prediction is, systematic error due to wrong
  asumption
  - Low bias: fewer assumptions, model match closely training dataset
  - High bias: more assumptions, model match less closely training dataset, can
    not capture the complexity, structure in the data
  - The lower the training error, the lower the bias. The higher the training
    error, the higher the bias.
- Variance: the amount of variability in the predictions, how much it can adjust
  to the new, unseen dataset
  - Low variance: the model is less sensitive to changes in the training data
    and can produce **consistent** estimates of the target function with
    different subsets of data from the same distribution
  - High variance: the model is very sensitive to changes in the training data
    and can result in significant changes in the estimate of the target function
    when trained on different subsets of data from the same distribution. Fit
    the **quirk** of the data you sample.
  - We want the variance to express how consistent a certain machine learning
    model is in its predictions when compared across similar datasets
  - When talking about the variance of a **particular** model, we always talk
    about one model, but **multiple** datasets. In practice, you would compare
    the model error on the training dataset and the error on the testing (or
    validation) dataset.

- Simple model: high bias, low variance
- Complex model: low bias, high variance

Weak learner: slightly better than random guessing (p > 0.5)

**Bagging** (Bootstrap Aggregating): run weak leaners independently from each
other in parallel and combines them following some kind of deterministic
averaging process

- Sample dataset with replacement into new multiple datasets (resamples or
  bootstrap samples)
- Use weak learner to train on each bootstrap
- Combine predictions using majority vote (classification) or average
  (regression)
- Does not reduce bias but reduce variance
- Naive mixture (all members weighted equally) -> can be improved by weighted
  ensembling

**Boosting**

- Train sequentially, each time focusing on what previous got wrong, focus its
  efforts on the most difficult observations
- Weighted training set, classifier “tries harder” on examples with higher cost

**AdaBoost** (Adaptive Boosting)

- At each iteration, re-weight the training samples by assigning larger weights
  to samples (data points) that were classified **incorrectly**
- Train a new base classifier based on the re-weighted samples
- Add it to the ensemble of classifiers with an appropriate weight
  - The better a weak learner performs, the more it contributes to the strong
    learner
- Repeat

### References

- [AdaBoost: A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://www.sciencedirect.com/science/article/pii/S002200009791504X)
- [Ensemble methods: bagging, boosting and stacking](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205)
- [Bias, Variance, and Overfitting Explained, Step by Step](https://machinelearningcompass.com/model_optimization/bias_and_variance/)
- [Random Forests for Complete Beginners](https://victorzhou.com/blog/intro-to-random-forests/)

## Genetic Algorithms (GA)

> A genetic algorithm (or GA) is a variant of stochastic beam search in which
> successor states are generated by combining two parent states rather than by
> modifying a single state.

GA is a search heuristic that will not stop unless we stop it by declare stop
condition.

Fitness: how good a solution is

- To find next generation
- Check if solution converge

Genetic function:

- Selection
  - Roulette wheel Selection (RWS): use circle to represent fitness, more
    fitness more area
  - Likely to choose from best fitness
- Crossover
- Mutation

### References

- [Lecture 13: Learning: Genetic Algorithms](https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/resources/lecture-13-learning-genetic-algorithms/)

## Reinforcement learning

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
- Value of a state is total amount reward the agent **hope** to accumulate in
  the future.
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

**Minimax**: explores all notes available. Hard to scale.

**Monte Carlo tree search (MCTS)**: combine tree search and random sampling.

> Essentially, MCTS uses Monte Carlo simulation to accumulate value estimates to
> guide towards highly rewarding trajectories in the search tree. In other
> words, MCTS pays more attention to nodes that are more promising, so it avoids
> having to brute force all possibilities which is impractical to do.

4 steps loop in constrained timd ans space:

- Selection: Using tree policy
  - Need to balance between exploration and exploitation
  - TODO: UCB1, UCT
- Expansion: add new node to the tree
- Simulation or Rollout: use default policy to produce outcome value
  - Choose random action until reach terminal state
- Backup or Backpropagation: re-update value of nodes along the line trace back
  to root
  - Update reward
  - Update visit count

**Markov Deision Processes (MDP)**

- Framework for planning and maximizing your future gains in scenarios where
  some things are out of your control but you still have choices to make.
- The future state only depends on the current state and the action taken, not
  on the history of previous states and actions

**Model-Based Learning**

### References

- [Monte Carlo Tree Search: An Introduction](https://towardsdatascience.com/monte-carlo-tree-search-an-introduction-503d8c04e168)
- [Monte Carlo Tree Search](https://www.youtube.com/watch?v=UXW2yZndl7U)

## Markov

Markove Chain: a sequence of random state where future state depends on current
state only.

## Meta learning and N-shot learning

**Meta learning**: learn to learn aka learn the learning process.

Why: Deep learning require large datasets.

2 phases:

- Meta-learning phase: Learn from a variety of tasks to get initial parameters.
- Adaption phase: Adapt to task-specific parameters.

Data:

- The meta-training: train examples (support set) and test examples (query set).
- The meta-validation and meta-test: same structure as meta-training.

**Few-shot learning (FSL)**: N-way-k-shot classification, to classify new sample
with only few training samples with labels.

- N: number of classes
- K: number of examples per class
- Each N has K samples

- Have: support set N-K
- Want: query Q new sample in which class ?

- Few way, more shot to improve accuracy

**One-shot learning (OSL)**: N-way-1-shot classification. Check similarity
between two images.

Detech camera face and passport face if same person.

**Zero-shot learning**: categorize new, unseen samples without any training
samples. Example: put corgi in dog class.

### References

- [🐣 From zero to research — An introduction to Meta-learning](https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a)
- [CS 182: Lecture 21: Part 1: Meta-Learning](https://www.youtube.com/watch?v=h7qyQeXKxZE)
- [Few-Shot Learning (1/3): Basic Concepts](https://www.youtube.com/watch?v=hE7eGew4eeg)

## Programming

### References

- [NumPy fundamentals](https://numpy.org/doc/stable/user/basics.html)
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
