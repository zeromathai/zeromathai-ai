# 04. Probabilistic Inference

A structured introduction to probabilistic reasoning in AI, covering uncertainty, Bayes' Theorem, probabilistic graphical models, Bayesian networks, and probabilistic inference under incomplete information.

---

## 📖 Full Article

- English  
https://zeromathai.com/en/probabilistic-reasoning-bayesian-network-en/

- Korean  
https://zeromathai.com/probabilistic-reasoning-bayesian-network/

👉 Other languages are available on the website

---

## 🧠 Overview

Probabilistic inference is one of the core ways AI handles uncertainty.

In real-world environments:

- observations are incomplete  
- sensors are noisy  
- causes are hidden  
- outcomes are not always deterministic  

Instead of treating knowledge as strictly true or false, probabilistic AI represents uncertainty numerically and updates beliefs as new evidence arrives.

---

## 📚 What This Section Covers

This lecture connects the following topics:

- uncertainty and probabilistic reasoning  
- rule-based systems vs probabilistic systems  
- probability and Bayes' Theorem  
- probabilistic graphical models  
- Bayesian networks  
- probabilistic inference methods  

---

## 🌫 Uncertainty and the Need for Probabilistic Reasoning

Most real-world AI problems involve uncertainty.

### Main sources of uncertainty

- **Stochastic environments**  
  the same action can produce different outcomes  

- **Partial observability**  
  the agent sees only part of the true world state  

- **Noise and ambiguity**  
  sensors and measurements are imperfect  

Probabilistic reasoning provides a framework for representing and updating uncertainty explicitly.

This is why it is central in areas such as:

- medical diagnosis  
- speech recognition  
- recommender systems  
- anomaly detection  
- fault diagnosis  

---

## ⚖ Rule-Based Expert Systems vs Probabilistic Systems

Early expert systems relied on symbolic IF–THEN rules.

### Rule-based reasoning
Example:

- IF fever AND cough THEN flu

This is useful for explicit and interpretable reasoning, but real environments often violate rigid rules.

### Probabilistic reasoning
Instead of asking only:

- Is this true or false?

probabilistic systems ask:

- What is the probability of this hypothesis given the evidence?

This makes them better suited to uncertain, noisy, and ambiguous environments.

Bayesian networks are one of the clearest examples of this shift from rigid symbolic rules to probabilistic models.

---

## 🎲 Probability and Bayes' Theorem

Probabilistic reasoning relies on several core ideas.

### Probability
A number between 0 and 1 expressing how likely an event is.

### Joint probability
The probability that two events happen together.

Example:
- it is raining **and** the road is slippery

### Marginal probability
The probability of one event regardless of the values of other variables.

### Conditional probability

P(A|B) = P(A ∩ B) / P(B)

This tells us how belief in **A** changes once **B** is known.

### Bayes' Theorem

P(H|E) = (P(E|H) * P(H)) / P(E)

Where:

- **H** = hypothesis  
- **E** = evidence  
- **P(H)** = prior belief  
- **P(E | H)** = likelihood  
- **P(H | E)** = posterior belief after observing evidence  

Bayes' Theorem is the basic rule for updating belief under uncertainty.

---

## 🕸 Probabilistic Graphical Models and Bayesian Networks

A probabilistic graphical model represents random variables and their dependencies through a graph.

A **Bayesian network** is a directed acyclic graph (DAG) in which:

- **nodes** represent random variables  
- **edges** represent direct dependencies or causal influence  
- **CPTs** (conditional probability tables) describe how each variable depends on its parent nodes  

### Key idea

Instead of modeling the full joint distribution directly, we factorize it into smaller conditional pieces.

This makes reasoning more structured and scalable.

### Why Bayesian networks matter

They allow AI systems to:

- reason from partial observations  
- represent causal structure  
- update hidden variables when evidence changes  

---

## 🔄 Probabilistic Inference in Bayesian Networks

Once a Bayesian network is defined, inference computes how probable unknown variables are given observed evidence.

### Main inference settings

- **Exact inference**  
  computes exact posterior probabilities  
  suitable for small or specially structured networks  

- **Approximate inference**  
  uses methods such as sampling when exact inference is too expensive  

- **MAP estimation**  
  finds the most probable explanation or variable assignment given evidence  

### Practical interpretation

Given evidence such as:

- a medical symptom  
- an alarm signal  
- a faulty sensor reading  

the system updates beliefs about hidden causes and chooses the most plausible explanation.

---

## 🔗 Related Topics

This section connects naturally to:

- Logical Inference  
- Search-Based Problem Solving  
- Bayesian Networks  
- Expert Systems  
- Decision Theory  
- Probabilistic Graphical Models  
- Uncertainty in AI  

---

## 💡 Why This Matters

Probabilistic inference is essential because real intelligence must work under uncertainty.

It allows AI to:

- represent incomplete knowledge  
- update beliefs with evidence  
- reason under ambiguity  
- support rational decision making  

This makes it one of the key foundations of modern AI.

---

## ⭐ Note

This material is organized and reconstructed based on lectures by **Professor Byoung-Tak Zhang (Seoul National University)**.

---

## 🚀 Next Steps

Future documents in this repository can expand this section into:

- Bayesian network structure learning  
- exact vs approximate inference algorithms  
- probabilistic decision making  
- graphical models beyond Bayesian networks
