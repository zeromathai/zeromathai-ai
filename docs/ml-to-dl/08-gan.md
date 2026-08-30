# 8. GAN

## 📌 Lecture Overview

Generative modeling aims to learn the underlying distribution of observed data so that a model can generate new samples with similar statistical structure. This perspective differs from discriminative modeling, which focuses primarily on predicting labels or outputs from inputs.

Understanding Generative Adversarial Networks (GANs) therefore requires a broader view of generative modeling: how data distributions are represented, how latent variables relate to observations, why posterior inference can become difficult, and why different deep generative models make different tradeoffs between tractability and flexibility.

A GAN approaches this problem through adversarial learning. Instead of explicitly evaluating a normalized probability density for every generated sample, it trains a generator and a discriminator against each other. The generator attempts to produce samples resembling real data, while the discriminator attempts to distinguish generated samples from real ones.

---

## 📖 Full Article

* English: [https://zeromathai.com/en/gan-en/](https://zeromathai.com/en/gan-en/)
* Korean: [https://zeromathai.com/gan/](https://zeromathai.com/gan/)

---

## 📚 Table of Contents

1. Discriminative and Generative Models
2. Generative Modeling
3. Forward and Inverse Problems
4. Posterior Inference
5. Approximate Inference
6. Data Distribution Modeling
7. Normalizing Constants
8. Deep Generative Models
9. Diffusion Models
10. GANs as Adversarial Generative Models

---

## Discriminative and Generative Models

A discriminative model focuses on the relationship between an observation and a target. Given an input $x$, it learns how to predict a corresponding label or output $y$.

A generative model asks a different question. Instead of learning only the decision boundary required for prediction, it attempts to model how observations themselves are distributed.

This distinction becomes especially important when the objective is not merely to classify existing data but to generate new data that follows the same underlying structure.

---

## Generative Modeling

Suppose observed samples come from an unknown data distribution $p_{\text{data}}(x)$. A generative model attempts to learn a model distribution $p_{\theta}(x)$ that approximates it.

The central objective can be expressed conceptually as

$$
p_{\theta}(x) \approx p_{\text{data}}(x).
$$

Once the model captures this distribution sufficiently well, new samples can be generated from the learned model:

$$
x \sim p_{\theta}(x).
$$

The difficulty is that real-world data distributions are usually high-dimensional and structurally complex. Images, for example, are not arbitrary collections of pixel values. Their probability mass is concentrated around structured configurations corresponding to meaningful objects, textures, shapes, and scenes.

Generative modeling therefore requires a model flexible enough to represent complex distributions while still allowing useful learning or sampling procedures.

---

## Forward and Inverse Problems

Many generative models can be understood through the relationship between latent variables and observations.

Let $z$ represent a latent variable and $x$ an observed variable. A forward generative process describes how an observation may be produced from a latent state:

$$
z \rightarrow x.
$$

If the model specifies a conditional distribution $p(x\mid z)$ together with a prior distribution $p(z)$, the joint distribution can be written as

$$
p(x,z)=p(x\mid z)p(z).
$$

The inverse problem goes in the opposite direction. Given an observation $x$, we want to infer which latent states $z$ could have produced it.

This requires the posterior distribution

$$
p(z\mid x).
$$

The forward process may be easy to describe while the corresponding inverse problem is difficult. This asymmetry is one of the central reasons inference becomes an important issue in probabilistic generative modeling.

---

## Posterior Inference

Bayes' rule gives the posterior distribution as

$$
p(z\mid x)
=
\frac{p(x\mid z)p(z)}{p(x)}.
$$

The denominator is the marginal probability of the observation:

$$
p(x)
=
\int p(x\mid z)p(z)\,dz.
$$

This term ensures that the posterior is properly normalized.

Although the equation is conceptually simple, computing the integral can become extremely difficult when $z$ is high-dimensional or when the underlying model is highly expressive.

As a result, exact posterior inference is often computationally intractable even when the generative process itself is easy to specify.

---

## Approximate Inference

When the exact posterior $p(z\mid x)$ cannot be computed efficiently, approximate inference methods attempt to obtain useful information without evaluating it exactly.

One major approach is Markov Chain Monte Carlo (MCMC), which constructs a Markov chain whose samples eventually approximate the target distribution.

Another approach is Variational Inference (VI). Instead of sampling directly from the difficult posterior, VI introduces a tractable distribution $q(z)$ and searches for the member of a distribution family $\mathcal{Q}$ that is closest to the true posterior.

The optimization can be written as

$$
q^\*(z)
=
\underset{q(z)\in\mathcal{Q}}{\arg\min}
D_{\mathrm{KL}}\!\left(
q(z)\,\|\,p(z\mid x)
\right).
$$

The KL Divergence is

$$
D_{\mathrm{KL}}\!\left(
q(z)\,\|\,p(z\mid x)
\right)
=
\mathbb{E}_{z\sim q(z)}
\left[
\log
\frac{q(z)}{p(z\mid x)}
\right].
$$

The key idea is that an intractable inference problem is converted into an optimization problem over a tractable family of distributions.

---

## Data Distribution Modeling

Generative models ultimately differ in how they represent and learn the data distribution.

Some models explicitly define a probability density $p_{\theta}(x)$ and optimize a likelihood-related objective. Others introduce latent variables and model a joint distribution such as $p_{\theta}(x,z)$. Still others avoid directly evaluating the likelihood and instead learn procedures capable of generating samples that resemble the data distribution.

This creates an important design question: how expressive can the model become while keeping probability computation, inference, or sampling manageable?

---

## Normalizing Constants

A valid probability distribution must be normalized.

For a model defined through an unnormalized function $\tilde{p}_{\theta}(x)$, the normalized probability can be written as

$$
p_{\theta}(x)
=
\frac{\tilde{p}_{\theta}(x)}{Z_{\theta}},
$$

where the normalizing constant is

$$
Z_{\theta}
=
\int \tilde{p}_{\theta}(x)\,dx.
$$

The model may be highly flexible, but computing $Z_{\theta}$ can become difficult or impossible in practice.

This illustrates a recurring tradeoff in probabilistic modeling. Greater representational flexibility can make normalization, likelihood evaluation, or inference computationally expensive.

---

## Deep Generative Models

Deep generative models use neural networks to represent complex data distributions or transformations associated with them.

Different model families resolve the tractability-flexibility problem in different ways.

Autoregressive models factorize a joint probability distribution into conditional probabilities. Variational Autoencoders introduce latent variables together with approximate inference. Normalizing Flows construct invertible transformations that permit exact density evaluation under suitable architectural constraints.

GANs take a different route. Rather than requiring explicit likelihood evaluation, they learn a generator through competition with a discriminator.

Diffusion Models use another strategy: they define a gradual noising process and learn how to reverse that process.

These approaches share the objective of learning complex data distributions, but the computational mechanisms used to reach that objective are fundamentally different.

---

## Diffusion Models

A Diffusion Model defines a forward process that gradually adds noise to data.

A typical forward transition can be represented as

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right).
$$

After sufficiently many steps, the original structure of the data is largely destroyed and the state approaches a simple noise distribution.

Generation requires reversing this process. A learned reverse model approximates transitions of the form

$$
p_{\theta}(x_{t-1}\mid x_t).
$$

Starting from noise, repeated reverse transitions gradually reconstruct a structured sample.

This provides a useful contrast with GANs. A GAN attempts to transform latent noise into a realistic sample through adversarial learning, whereas a Diffusion Model learns a sequence of denoising transitions.

---

## GANs as Adversarial Generative Models

A GAN contains two competing neural networks.

The **Generator** $G$ receives a latent variable $z$ sampled from a prior distribution and produces a generated sample:

$$
z \sim p_z(z),
\qquad
x_{\text{fake}} = G(z).
$$

The **Discriminator** $D$ receives either real or generated data and attempts to determine whether the input came from the real data distribution.

The original adversarial objective is

$$
\min_G \max_D V(D,G)
=
\mathbb{E}_{x\sim p_{\text{data}}(x)}
\left[
\log D(x)
\right]
+
\mathbb{E}_{z\sim p_z(z)}
\left[
\log\left(1-D(G(z))\right)
\right].
$$

The discriminator improves by assigning high values to real samples and low values to generated samples. The generator improves by producing samples that make this discrimination increasingly difficult.

The resulting training process can be viewed as a two-player game. Ideally, the generator distribution approaches the data distribution:

$$
p_g(x)=p_{\text{data}}(x).
$$

At this point, the discriminator can no longer reliably distinguish generated samples from real ones.

GANs therefore provide a way to learn a generative mechanism without requiring the model to explicitly calculate a normalized likelihood for each observation. This flexibility is one of their central strengths, but adversarial optimization also makes training sensitive to the interaction between the generator and discriminator.

---

## 🎯 Key Insight

The central problem of generative modeling is not simply generating an output that looks plausible. The deeper objective is to learn enough of the structure of the underlying data distribution to produce new samples consistent with it.

Posterior inference, approximate inference, normalization, latent-variable modeling, adversarial learning, and diffusion can all be understood as different responses to the computational difficulties created by this objective.

GANs are distinctive because they avoid direct likelihood evaluation and instead train a generator through an adversarial discriminator. Their importance becomes clearer when viewed as one solution within the broader landscape of deep generative modeling.

---

## 📚 Related Advanced Topics

* [Discriminative vs Generative Models](https://zeromathai.com/en/discriminative-vs-generative-models-en/)
* [Generative Modeling](https://zeromathai.com/en/generative-modeling-en/)
* [Inverse Problems](https://zeromathai.com/en/inverse-problems-en/)
* [Posterior Inference](https://zeromathai.com/en/posterior-inference-course-en/)
* [Approximate Inference](https://zeromathai.com/en/approximate-inference-course-en/)
* [Data Distribution Modeling](https://zeromathai.com/en/data-distribution-modeling-en/)
* [Normalizing Constant](https://zeromathai.com/en/normalizing-constant-concept-en/)
* [Deep Generative Models](https://zeromathai.com/en/deep-generative-models-en/)
* [Diffusion Models](https://zeromathai.com/en/diffusion-models-en/)

---

## ⭐ Note

This GitHub version is adapted from the full ZeroMathAI lecture for repository-friendly reading.

---

## 🔗 Navigation

This is the only lecture in the current set.
