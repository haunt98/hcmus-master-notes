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
  - **Partial observation** of state of the world.
  - Then pick one of possible actions -> let environment change from one state
    to another.
- Learning by adapting behavior based on collected rewards.
- Reaching goal by maximizing rewards.

Concepts:

- State `s` is a **complete** description of state of the world.
- Observation `o` is a **partial** description of a state.
  - Fully or partially observable environment: like chess or poke.
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
