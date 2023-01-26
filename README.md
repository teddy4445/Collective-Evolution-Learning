# Collective Evolution Learning Model for Vision-Based Collective Motion with Collision Avoidance (Code name: Hagabubim)
A simulator of a collective motion with collision avoidance using two-dimentional visual input performed by a novel genetic-based algorithms that extrapolates swarm behavior from individual behavior.   

The name **"Hagabubim"** is a fusion of two words: "hagav" and "habob". The first is the Hebrew word for "locust" and the latter is the Arabic word for "friend" which is commonly used in Hebrew speaking individuals as well.
Therefore, the name "Hagabub" stands for a friendly locus and "Hagabubim" is the plural form with the Hebrew postfix "im" which means "friendly locusts".

### Table of Contents
1. [Abstract](#abstract)     
2. [Getting Started](#usage)
3. [dependancies](#dependancies)
4. [How to cite](#how)


<a name="abstract"/>

## Abstract
Collective motion (CM) takes many forms in nature; schools of fish, flocks of birds, and swarms of locusts to name a few. Commonly, during CM the individuals of the group avoid collisions. These CM and collision avoidance (CA) behaviors are based on input from the environment such as smell, air pressure, and vision, all of which are processed by the individual and defined action. In this work, a novel vision-based CM with CA model (i.e., VCMCA) simulating the collective evolution learning process is proposed. In this setting, a learning agent obtains a visual signal about its environment, and throughout trial-and-error over multiple attempts, the individual learns to perform a local CM with CA which emerges into a global CM with CA dynamics. The proposed algorithm was evaluated in the case of locusts' swarms, showing the evolution of these behaviors in a swarm from the learning process of the individual in the swarm. Thus, this work proposes a biologically-inspired learning process to obtain multi-agent multi-objective dynamics.

<a name="usage"/>

## Getting Started

1. Clone the repo
2. Setup a Python 3.7 virtual endearment.
3. Install the 'requirements.txt' file (pip install requirements.txt)
4. run the project from **main.py** in order to train \ test \ test the simulation.  
5. run the project from **experiments_and_papers.py** to re-produce the results shown in the manuscript and others.

<a name="algorithm"/>

## Algorithm 
The CM behavior of locusts is believed to develop over time throughout an evolution process. The locusts try to adapt to their environment and generate the next population of locusts. During this process, multiple locusts experienced similar situations and operate slightly differently based on unique properties each locust has which resulted in different outcomes. Locusts that perform better were more likely to reproduce and keep their behavior in the population over time. As such, the wanted behavior (e.g., CM with CA) emerged throughout the collective experience of the locust and improved over time using a natural trial-and-error approach.

Based on this motivation, we propose a reinforcement learning (RL) based algorithm called Collective Evolution Learning (CEL for short). The proposed algorithm is based on the assumption the agents in a swarm are identical and share common knowledge (policy). Intuitively, the ECL algorithm receives the observable parameters of a subset of agents in the local environment of an agent and returns the action that best approximate the optimal action of the CM with CA objective metric using a Q-learning based model. Since both CM and CA are swarm-level metrics, ECL has an intermediate genetic algorithm (GA) based model to optimize CM and CA on the agent's level. Moreover, to obtain a well-performing Q-learning-based model, one is required to well sample the state space of the model. In order to do that, ECL introduces a layer on top of the Q-learning model that measures its performance for different scenarios and uses the K-nearest neighbors (KNN) algorithm to define a new sampling strategy for further training of the Q-learning model. 

More details are avalible in the paper and the code.

<a name="dependancies"/>

## Dependencies 
- Python            3.7
- numpy             1.16.1
- matplotlib        3.4.1
- pandas            0.24.1
- sklearn-learn     0.23.0
- seaborn           0.11.2
- scipy             Latest

These can be found in the **requirements.txt** and easily installed using the "pip install requirements.txt" command in your terminal. 

<a name="how"/>

## How to cite
Please cite the EMPSR work if you compare, use, or build on it:
```
@article{lazebnik2022,
  title={Collective Evolution Learning Model for Vision-Based Collective Motion with Collision Avoidance},
  author={Krongauz, D. and Lazebnik, T.},
  journal={bioRxiv},
  year={2022}
}
```

## Authors
Lazebnik, Teddy ([lazebnik.teddy@gmail.com](lazebnik.teddy@gmail.com)) 
Krongauz, David ([kingkrong@gmail.com](kingkrong@gmail.com))

## License
This project is licensed under the MIT1 License - see the LICENSE.md file for details.
