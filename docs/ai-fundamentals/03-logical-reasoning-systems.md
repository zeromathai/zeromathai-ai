# 03. Logical Reasoning Systems

A structured introduction to logical reasoning in AI, covering knowledge representation, propositional logic, first-order logic, inference rules, logic programming, and theorem proving.

---

## 📖 Full Article

- English  
https://zeromathai.com/en/ai-logical-reasoning-systems-en/

- Korean  
https://zeromathai.com/ai-logical-reasoning-systems/

👉 Other languages are available on the website

---

## 🧠 Overview

Logical reasoning systems aim to represent knowledge in a precise form and derive conclusions step by step using formal rules.

Instead of relying on intuition, logic-based AI:

- encodes knowledge as formulas  
- applies inference rules  
- derives conclusions systematically  

---

## 📚 What This Section Covers

This lecture connects the following topics:

- logic and knowledge representation  
- propositional logic  
- first-order logic  
- inference rules  
- logic programming (PROLOG)  
- theorem proving  
- commonsense knowledge representation  

---

## 🧩 Logic and Knowledge Representation

The first question in logical AI is:

👉 **How do we represent what we know?**

Natural language is ambiguous, so we convert knowledge into logical form:

- facts → logical symbols  
- rules → implications  
- collection of facts and rules → **knowledge base (KB)**  

This allows machines to reason precisely.

---

## 🔲 Propositional Logic

Propositional logic works with statements that are either **true or false**.

### Key components

- logical operators: AND, OR, NOT  
- implication: A → B  
- truth tables  

### Example

If:

- A = “it is raining”  
- B = “the road is slippery”  

Then:

- A → B  
- A is true → B is true  

👉 This is the simplest form of logical inference.

---

## 🧠 First-Order Logic (FOL)

First-order logic extends propositional logic by representing:

- objects  
- properties  
- relationships  

### Key elements

- predicates: `Human(x)`, `Loves(x, y)`  
- variables and constants  
- quantifiers:
  - ∀ (for all)  
  - ∃ (there exists)  

### Example

∀x (Student(x) -> Human(x))

👉 FOL allows AI to describe real-world relationships precisely.

---

## 🔄 Inference in First-Order Logic

Inference means deriving new knowledge from a knowledge base.

Example:

- All students are human  
- Minsu is a student  

👉 Therefore: Minsu is human  

This requires:

- substitution  
- rule application  
- structured reasoning  

---

## 🏗 Logical Modeling of the World

Real-world situations can be encoded as logical structures.

### Example: Blocks world

- `On(B, A)`  
- `Clear(B)`  
- `Floor(A)`  

### Example: Family relations

Logical rules can infer:

👉 "Tom is Brian’s grandfather"  

from parent relationships.

---

## ⚙️ Inference Rules

Inference rules define how knowledge is extended safely.

### Modus Ponens

(A → B), A ⇒ B

### And-Elimination

A ∧ B ⇒ A, B

### Resolution

- removes contradictions  
- combines complementary statements  

### First-order extensions

- Universal Instantiation (UI)  
- Existential Generalization (EG)  

👉 These rules allow step-by-step reasoning.

---

## 💻 PROLOG and Logic Programming

PROLOG is a logic-based programming language.

Instead of instructions, it uses:

- **Facts**
- **Rules**
- **Queries**

### Example

```prolog
bat_ok.
liftable.
moves :- bat_ok, liftable.
