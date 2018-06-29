# Eligibility State Representation
This repository contains the experiments I perform with Eligibilty State Representation.
Eligibilty State Representation is a simple idea of summing up the discounted observations over time in POMDP to form a state. A linear function approximation is built on top of this state representation to estimate a value function.

If $O_t$ is the observation over time $t$. Then the state $S-t$ at any time step is represented as,
\begin{equation}
S_t = \sum_{i=0}^{t} O_i * \lambda^{t-i}
\end{equation} 
  
## Requirements
This package requires,
* python >= 3.6
* gym >= 0.10.5
* numpy >= 1.14.3
* pytorch >= 0.4.0

## To run the code
Run the below command to execute the code,
```
python main.py --env=ENV_NAME --eps=INITIAL_EXPLORATION --n_epi=NUM_EPISODES --lr_rate=LEARNING_RATE --trace=TRACE_PARAMETER --gamma=DISCOUNT --n_exp=NUM_EXPERIMENTS --FA_type=TYPE_OF_FUNCTION_APPROXIMATOR
```
The package currenlty supports,
* env - Any OpenAI GYM environments
* FA_type - Linear Function Approximator (OneLinearNet)