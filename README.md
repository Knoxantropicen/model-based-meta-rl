# Model-Based Meta-Reinforcement Learning
A PyTorch implementation of paper [Learning to Adapt in Dynamic, Real-World Environments through Meta-Reinforcement Learning](https://arxiv.org/pdf/1803.11347.pdf).

## Todo List
- Achieve good performance using purely model-based RL (debug mode)
- Fix gradient explosion problem caused by multiple steps of adaptation update (higher order gradient issues)
- Debug NLL loss
- Hyper-parameter tuning
- Improve MAML stability by [MAML++](https://arxiv.org/pdf/1810.09502.pdf)
- More parallelization (async evaluation, parallel env...)
