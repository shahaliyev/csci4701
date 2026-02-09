---
description: >
  Information theory provides the mathematical language for measuring uncertainty, surprise, and the difference between probability distributions. This page introduces the core information-theoretic concepts used in deep learning, including self-information, entropy, cross-entropy, and KL divergence.
---

# Information Theory

<div style="margin:.3rem 0 1rem;font-size:.9em;color:#555;display:flex;align-items:center;gap:.35rem;font-family:monospace">
  <time datetime="2026-02-09">9 Feb 2026</time>
</div>

[Information theory](https://en.wikipedia.org/wiki/Information_theory) is the mathematical framework for measuring how much *uncertainty* or *information* is contained in a probability distribution. While [probability theory](../03_probability) tells us how to represent uncertainty, information theory tells us how to *quantify* it. In [deep learning](../../introduction/01_deep_learning), information theory appears everywhere: in loss functions, in [regularization](../../notebooks/04_regul_optim), in probabilistic modeling ([VAEs](../../notebooks/07_vae)), and in the general idea of compressing data into meaningful representations.

!!! info
    The following source was consulted in preparing this material: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Chapter 3: Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html).

!!! warning "Important"
    Some concepts in this material are simplified for pedagogical purposes. These simplifications slightly reduce precision but preserve the core ideas relevant to deep learning.

!!! note
    Information theory was established by [Claude Shannon](https://en.wikipedia.org/wiki/Claude_Shannon) in his famous 1948 paper [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), where he introduced the idea that information can be measured quantitatively, just like mass or energy. Shannon originally developed the theory to study how efficiently messages can be transmitted through noisy communication channels (e.g. radio). Today, the same mathematical tools are fundamental in deep learning, because training a model often means minimizing uncertainty and compressing information into useful representations.
## Self-Information

The core idea of information theory is simple: _Learning that an unlikely event happened gives more information than learning that a likely event happened._ 

!!! example
    Learning that "the sun rose today" is not informative, but learning that "a solar eclipse happened today" is informative.

To measure this idea mathematically, we want a quantity that behaves like **surprise**:

- If an event is very likely, it should have low surprise.
- If an event is rare, it should have high surprise.

If two independent events occur, their probabilities multiply, so their surprise would multiply as well. Similar to [log-likelihood](../05_prob_modeling), to convert multiplication into addition, we can take the logarithm.

!!! warning "Important"
    Multiplying probabilities quickly produces extremely small numbers. For example, if we observe $100$ independent events each with probability $0.01$, then the joint probability $p(x_{1:100})$ is:
    $$
    (0.01)^{100} = 10^{-200},
    $$
    which is essentially zero in floating-point arithmetic.   This causes numerical underflow and also leads to very small gradients when optimizing likelihoods directly. Taking the logarithm fixes this problem:
    $$
    \log (0.01^{100})
    =
    100 \log(0.01)
    \approx
    -460.5,
    $$
    which is a normal-sized number. This is why deep learning almost always optimizes log-likelihood instead of likelihood: products become sums, computations stay stable, and gradients remain usable.


Using the identity $\log(1/u)=-\log(u)$, we obtain the [self-information](https://en.wikipedia.org/wiki/Information_content) (also called *surprisal*, *information content*, or *Shannon information*) of an event $X=x$:
$$
I(x) = -\log P(X=x).
$$

This definition satisfies three key properties defined by Shannon:

1. Certain events contain no information: if $P(x)=1$, then $I(x)=0$.
2. Unlikely events contain more information: smaller $P(x)$ gives larger $I(x)$.
3. Independent information is additive: if events are independent, their probabilities multiply, so their information adds.

!!! note
    In this course, $\log$ always means the natural logarithm (base $e$). When using natural logs, information is measured in _nats_ rather than bits.

## Entropy

Self-information measures the surprise of a single outcome. But often we want a single number that summarizes the uncertainty of an entire distribution. In information theory, the (Shannon) [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of a discrete random variable $X$ is the expected self-information:
<div style="overflow-x:auto; max-width:100%;">
$$
H(X)
=
\mathbb{E}[I(X)]
=
-\sum_x P(X=x)\log P(X=x).
$$
</div>

It implies that if $X$ is almost always the same value, there is little uncertainty, so entropy is low. If $X$ has many equally likely outcomes, uncertainty is high, so entropy is high. Discrete entropy is always nonnegative: $H(X)\ge 0.$

!!! note
    Entropy is maximized by the uniform distribution. If $X$ takes $k$ outcomes with equal probability $P(X=i)=1/k$, then:
    $$
    H(X)
    =
    -\sum_{i=1}^{k}\frac{1}{k}\log\frac{1}{k}
    =
    \log k.
    $$
    This matches intuition: choosing among more equally likely options is more uncertain.

!!! example
    Consider a Bernoulli random variable $X\sim \mathrm{Bernoulli}(\phi)$:
    $$
    P(X=1)=\phi,\qquad P(X=0)=1-\phi.
    $$
    Its entropy is:
    $$
    H(X)
    =
    -\phi\log\phi-(1-\phi)\log(1-\phi).
    $$
    This entropy is highest at $\phi=0.5$ (maximum uncertainty) and approaches $0$ as $\phi\to 0$ or $\phi\to 1$ (almost no uncertainty).

For a continuous random variable with density $p(x)$, the analogous quantity is:
$$
H(X)
=
-\int p(x)\log p(x)\,dx.
$$

[Differential (continuous) entropy](https://en.wikipedia.org/wiki/Differential_entropy) measures uncertainty in a continuous distribution, but it behaves differently from discrete entropy. This happens because $p(x)$ is a probability *density*, not a probability. A density can be greater than $1$, so $\log p(x)$ can be positive. Hence, differential entropy can be negative.

Another important difference is that differential entropy depends on the *units* of measurement. If we scale a continuous variable, its differential entropy changes. If $Y=aX$ for some constant $a>0$, then:
$$
H(Y)=H(X)+\log a.
$$

!!! note
    This means that differential entropy is not an absolute measure of "how much uncertainty exists." Changing meters to centimeters changes the value.

!!! example
    Let $X\sim U(0,1)$, so $p(x)=1$ on $[0,1]$. Then:
    $$
    H(X)
    =
    -\int_0^1 1\cdot \log(1)\,dx
    =
    0.
    $$

    Now let $Y\sim U(0,0.1)$, so $p(y)=10$ on $[0,0.1]$. Then:
    $$
    H(Y)
    =
    -\int_0^{0.1} 10\log(10)\,dy
    =
    -\log(10),
    $$
    which is negative.

## Conditional Entropy

Sometimes uncertainty is reduced after observing another variable. The [conditional entropy](https://en.wikipedia.org/wiki/Conditional_entropy) measures the remaining uncertainty in $X$ after we know $Y$:
<div style="overflow-x:auto; max-width:100%;">
$$
H(X \mid Y)
=
-\sum_{x,y} P(x,y)\log P(x \mid y).
$$
</div>

Equivalently, it is the expected entropy of the conditional distribution:
<div style="overflow-x:auto; max-width:100%;">
$$
H(X \mid Y)
=
\sum_y P(y)\,H(X \mid Y=y).
$$
</div>

If $X$ is completely determined by $Y$, then $H(X\mid Y)=0$. If $X$ and $Y$ are independent, then observing $Y$ gives no information about $X$, so $H(X\mid Y)=H(X)$.



## Cross-Entropy

Entropy $H(P)$ measures the uncertainty of a distribution $P$. In learning, we usually have two distributions:

- The *true* (data) distribution $P$ that generates labels.
- A *model* distribution $Q$ that tries to predict them.

[Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) measures the expected number of _nats_ needed to encode outcomes generated by the true distribution $P$, when we use an encoding scheme (or predictive model) that assumes the distribution is $Q$. In other words, it measures how well $Q$ predicts samples coming from $P$. If $Q$ assigns low probability to events that happen frequently under $P$, the cross-entropy becomes large.
$$
H(P,Q)
=
-\sum_x P(x)\log Q(x).
$$
Compare this with entropy:
$$
H(P) = -\sum_x P(x)\log P(x).
$$

The only difference is what appears inside the logarithm: entropy uses the true distribution $P$, while cross-entropy uses the model distribution $Q$.

!!! example
    Suppose $X$ represents the outcome of a biased coin, and the true distribution is:
    $$
    P(X=\text{heads}) = 0.9,
    \qquad
    P(X=\text{tails}) = 0.1.
    $$

    The entropy of the true distribution is:
    <div style="overflow-x:auto; max-width:100%;">
    $$
    H(P)
    =
    -0.9\log(0.9) - 0.1\log(0.1)
    \approx
    0.325 \text{ nats}.
    $$
    </div>
    Now suppose we build a wrong model $Q$ that assumes the coin is fair:
    $$
    Q(X=\text{heads}) = 0.5,
    \qquad
    Q(X=\text{tails}) = 0.5.
    $$

    The cross-entropy of $P$ relative to $Q$ is:
    <div style="overflow-x:auto; max-width:100%;">
    $$
    H(P,Q)
    =
    -0.9\log(0.5) - 0.1\log(0.5)
    =
    -\log(0.5)
    =
    \log 2
    \approx
    0.693 \text{ nats}.
    $$
    </div>

    So even though the true coin has relatively low uncertainty ($H(P)\approx 0.325$), using the wrong model distribution $Q$ increases the expected code length to $H(P,Q)\approx 0.693$. 
    
    Intuitively, this happens because the model assigns too little probability to the outcome that occurs most of the time (heads). Cross-entropy penalizes this mismatch. If we instead choose $Q=P$, then:
    $$
    H(P,P)=H(P),
    $$
    meaning cross-entropy becomes minimal when the assumed distribution matches the true one.

## Kullback-Leibler Divergence

Cross-entropy answers a practical question: *If the world follows $P$, but we build a model as if it were $Q$, how costly is that mistake on average?* But sometimes we want a cleaner question: _How much worse is $Q$ compared to the true distribution $P$?_

The answer is the extra penalty we pay when we use $Q$ instead of $P$.
This difference is exactly the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):
$$
D_{\mathrm{KL}}(P\|Q)
=
H(P,Q)-H(P).
$$

Equivalently, it can be written as:
$$
D_{\mathrm{KL}}(P\|Q)
=
\sum_x P(x)\log\frac{P(x)}{Q(x)}.
$$

Or for continuous distributions:
$$
D_{\mathrm{KL}}(P \| Q)
=
\int p(x)\log \frac{p(x)}{q(x)}\,dx.
$$

So KL divergence measures the *gap* between cross-entropy and entropy: it is zero when $P=Q$, and grows as $Q$ becomes a worse approximation of $P$. 

!!! note
    KL divergence is always nonnegative:
    $$
    D_{\mathrm{KL}}(P\|Q)\ge 0,
    $$
    and equals $0$ if and only if $P=Q$ ([almost everywhere](../03_probability/#measure-theory) in the continuous case).


!!! warning "Important"
    KL divergence is not symmetric:
    $$
    D_{\mathrm{KL}}(P \| Q) \ne D_{\mathrm{KL}}(Q \| P).
    $$
    So it is not a true distance metric, but it is still one of the most important ways to compare distributions in deep learning. 

A key identity connects entropy, cross-entropy, and KL divergence:
$$
H(P,Q)
=
H(P) + D_{\mathrm{KL}}(P \| Q).
$$

This immediately explains why minimizing cross-entropy makes sense. Since $H(P)$ does not depend on the model $Q$, minimizing $H(P,Q)$ over $Q$ is equivalent to minimizing KL divergence:
$$
\arg\min_Q H(P,Q)
=
\arg\min_Q D_{\mathrm{KL}}(P\|Q).
$$

!!! note
    In classification, minimizing cross-entropy is the same as minimizing $D_{\mathrm{KL}}(P\|Q)$ between the true label distribution and the model predictions.

!!! success "Exercise"
    Let $P=\mathcal{N}(\mu_1,\sigma_1^2)$ and $Q=\mathcal{N}(\mu_2,\sigma_2^2)$. Derive a [closed-form expression](https://en.wikipedia.org/wiki/Closed-form_expression) for $D_{\mathrm{KL}}(P\|Q)$ in terms of $\mu_1,\mu_2,\sigma_1,\sigma_2$.


## Mutual Information

Entropy measures uncertainty in a single random variable. [Mutual information](https://en.wikipedia.org/wiki/Mutual_information) measures how strongly two random variables are related: how much knowing one reduces uncertainty about the other. Between $X$ and $Y$, it is defined as:
$$
I(X;Y)
=
\sum_{x,y} P(x,y)\log\frac{P(x,y)}{P(x)P(y)}.
$$

For continuous variables:
$$
I(X;Y)
=
\int p(x,y)\log\frac{p(x,y)}{p(x)p(y)}\,dx\,dy.
$$

!!! note
    Mutual information compares the true joint distribution $P(X,Y)$ with what the joint distribution would look like if $X$ and $Y$ were independent. If $X$ and $Y$ are independent, then $P(X,Y)=P(X)P(Y)$, so $I(X;Y)=0$.

Mutual information can be written as a KL divergence:
$$
I(X;Y)
=
D_{\mathrm{KL}}\big(P(X,Y)\,\|\,P(X)P(Y)\big).
$$

It can also be expressed using entropy:
<div style="overflow-x:auto; max-width:100%;">
$$
I(X;Y)
=
H(X)-H(X\mid Y)
=
H(Y)-H(Y\mid X).
$$
</div>

!!! note
    This shows that the information is the reduction in uncertainty of $X$ after observing $Y$ (and vice versa). It is always nonnegative: $I(X;Y)\ge 0$.