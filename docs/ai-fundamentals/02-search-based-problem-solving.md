# 02. Search-Based Problem Solving

A structured introduction to search-based problem solving, covering state spaces, search trees, heuristics, optimal search, local search, adversarial search, and search in complex environments.

---

## 📖 Full Article

- English  
https://zeromathai.com/en/ai-search-based-problem-solving-en/

- Korean  
https://zeromathai.com/ai-search-based-problem-solving/

👉 Other languages are available on the website

---

## 🧠 Overview

Search-based problem solving treats a problem as a space of possible states and actions, then tries to find a path from an initial state to a goal state.

This framework helps organize many different AI search algorithms into one coherent view.

---

## 📚 What This Section Covers

This lecture connects the following topics:

- problem formulation and state space representation  
- search trees and graph-search structure  
- uninformed search  
- heuristic (informed) search  
- A* search and evaluation functions  
- local search and optimization  
- adversarial search and game search  
- search in complex environments  
- heuristic quality, search cost, and performance factors  

---

## 🔍 Problem Formulation and the State Space Model

Search-based problem solving begins by expressing a problem structurally.

A problem is represented through:

- **State** — a description of the world at a given moment  
- **Initial state** — where search starts  
- **Actions / operators** — possible moves or transformations  
- **Transition model** — how actions change states  
- **Goal test** — whether the goal condition is satisfied  
- **Path cost** — the cumulative cost of a path  

A good formulation makes search easier, while a poor formulation can make even simple problems difficult.

---

## 🌳 Search Trees and Graph Search

Search generates possible states step by step.

Two common structural views are:

- **Search tree** — expands nodes from the initial state as a tree  
- **Graph search** — records visited states to avoid revisiting and infinite loops  

Important concepts include:

- **Branching factor (b)** — average number of successors  
- **Depth (d)** — depth of the shallowest goal  
- **Time complexity** — how much computation is needed  
- **Space complexity** — how much memory is required  

As the state space grows, search can become exponentially expensive.

---

## 🚶 Uninformed Search

Uninformed search uses no extra hint about where the goal is.

Main methods include:

- **Breadth-First Search (BFS)**  
  Explores all nodes at the current depth first.  
  With uniform step costs, it finds the shortest path.

- **Depth-First Search (DFS)**  
  Explores one branch as deeply as possible.  
  Uses little memory, but may be inefficient or get lost in deep spaces.

- **Iterative Deepening Search (IDS)**  
  Combines BFS-style completeness with DFS-style memory efficiency by repeatedly increasing the depth limit.

These methods are often compared in terms of:

- completeness  
- optimality  
- time complexity  
- space complexity  

---

## 🧭 Heuristic Search

Heuristic search improves efficiency by estimating how close a state is to the goal.

### Heuristic function
- **h(n)** estimates the remaining cost from state **n** to a goal

### Greedy Best-First Search
- always expands the node with the smallest **h(n)**  
- often fast, but not always optimal  

### Evaluation function
- determines priority in the search queue  
- acts as the decision rule for which node to expand next  

A good heuristic should be informative without being too expensive to compute.

---

## ⭐ A* Search and the Evaluation Function

A* combines the actual path cost so far with an estimate of the remaining cost.

\[
f(n) = g(n) + h(n)
\]

Where:

- **g(n)** = real accumulated cost from the start to node **n**  
- **h(n)** = estimated remaining cost from **n** to the goal  
- **f(n)** = total estimated solution cost through **n**  

A* is especially important because it can guarantee an optimal solution when the heuristic is **admissible**, meaning it never overestimates the true remaining cost.

---

## 📈 Local Search and Optimization

Local search does not keep full paths in memory.

Instead, it works with the current state and its neighbors, trying to improve the solution step by step.

Main methods include:

- **Hill Climbing**  
  Repeatedly moves to a better neighboring state

- **Simulated Annealing**  
  Sometimes accepts worse moves to escape local optima

- **Local Beam Search / Genetic Algorithms**  
  Maintains multiple candidates and improves them in parallel

This family of methods is especially useful in very large or continuous spaces.

In continuous optimization problems, methods such as:

- **Gradient Descent**
- **Newton’s Method**

can also be understood as forms of local search.

This is why neural network training can be interpreted as local search in a high-dimensional continuous space.

---

## ♟ Adversarial Search and Game Search

Adversarial search applies when multiple agents compete.

Typical examples include chess, Go, and tic-tac-toe.

Main concepts include:

- **Game tree** — expands possible moves and opponent responses  
- **Minimax search** — assumes the opponent responds optimally  
- **Alpha–Beta pruning** — removes branches that cannot affect the final decision  

This connects search to:

- game theory  
- decision theory  
- reinforcement learning  

---

## 🌫 Search in Complex Environments

Real-world environments are often not fully observable, deterministic, or static.

Important extensions include:

### Nondeterministic search
- the same action may lead to multiple possible outcomes  
- often represented using **AND–OR trees**

### Partially observable search
- the exact state is unknown  
- the agent reasons over a **belief state**

### Online search
- the state space or transition model is unknown in advance  
- planning and execution must happen together  

These ideas lead naturally toward more advanced topics such as:

- reinforcement learning  
- POMDPs  
- exploration in unknown environments  

---

## ⚖️ Heuristic Quality, Search Cost, and Performance Factors

Search performance depends heavily on:

- branching factor  
- depth  
- duplicate states  
- heuristic quality  
- memory usage  

Key evaluation criteria are:

- **Completeness** — will it find a solution if one exists?  
- **Optimality** — will it find the best solution?  
- **Time complexity** — how much computation does it require?  
- **Space complexity** — how much memory does it require?  

These criteria are not limited to search alone. They also connect search-based thinking to planning, reasoning, and reinforcement learning.

---

## 🔗 Related Topics

This section connects naturally to:

- State Space Representation  
- Heuristic Search  
- A* Search  
- Local Search  
- Optimization  
- Adversarial Search  
- Game Trees  
- Online Search  
- Reinforcement Learning  
- POMDPs  

---

## 💡 Why This Matters

Search-based problem solving provides one of the clearest structural views in AI.

It shows how many seemingly different algorithms can be understood as variations of the same core idea:

- represent states  
- define actions  
- search for a path or solution  
- balance cost, speed, and memory  

---

## ⭐ Note

This material is organized and reconstructed based on lectures by **Professor Byoung-Tak Zhang (Seoul National University)**.

---

## 🚀 Next Steps

Future documents in this repository can expand this section into more detailed subtopics such as heuristic search, local optimization, and adversarial search.
