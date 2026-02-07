# Probability Theory

<div style="margin:.3rem 0 1rem;font-size:.9em;color:#555;display:flex;align-items:center;gap:.35rem;font-family:monospace">
  <time datetime="2026-02-07">7 Feb 2026</time>
</div>

<div class="admonition warning">
  <p class="admonition-title">Important</p>
  <p style="margin: 1em 0;">
    The page is currently under development.
  </p>
</div>

Probability theory is the mathematical framework for reasoning under uncertainty. In artificial intelligence, probability is used in two main ways: (i) as a guide for how an intelligent system should reason under uncertainty, and (ii) as a tool for analyzing and understanding the behavior of learning algorithms. [Deep learning](../../introduction/01_deep_learning) relies on probability because real-world data is noisy, incomplete, and never fully deterministic, so uncertainty is unavoidable.  

!!! info
    The following source was consulted in preparing this material: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Chapter 3: Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html).

!!! warning "Important"
    Some concepts in this material are simplified for pedagogical purposes. These simplifications slightly reduce precision but preserve the core ideas relevant to deep learning.

## Kolmogorov Axioms

Probability is a consistent system governed by a small set of rules (axioms) that prevent contradictions. In modern mathematics, these rules are usually given by the axioms introduced by Kolmogorov[^kolmogorov]. The three axioms define probability as a function $P(\cdot)$ that assigns a number to each event in a set of valid events. For any event $A$ and $B$, a probability function must satisfy the following basic rules:

- It can never assign a negative value: $P(A) \ge 0$
- It must assign probability $1$ to a certain event: $P(\text{certain event}) = 1$
- If two events cannot happen at the same time ($A \cap B = \emptyset$), their probabilities add up: $P(A \cup B)=P(A)+P(B)$.

These axioms guarantee that probability remains logically consistent.[^dutchbook] Probability assigns numbers between $0$ and $1$ to events in order to represent how plausible those events are, given some assumptions or information. A value of $0$ means the event is impossible under the assumed model, a value of $1$ means it is certain, and intermediate values represent partial uncertainty.

## Two Views of Probability

Historically, probability was first developed to describe repeatable experiments, such as rolling dice, drawing cards, or observing outcomes in games of chance. Under this interpretation, called **frequentist probability**, $P(A)$ represents the long-run fraction of times event $A$ occurs if the experiment were repeated infinitely many times. For example, $P(\text{heads})=0.5$ means that if we toss a fair coin a very large number of times, about half of the tosses will result in heads.[^frequentist]

Later, probability began to be used in a broader sense: not only for repeatable experiments, but also for reasoning about unique situations where repetition is impossible. For example, a doctor may assign a probability that a patient has a disease, even though we cannot create infinitely many identical copies of the patient. In this interpretation, called **Bayesian probability**, probability measures a _degree of belief_ given incomplete information.

Although the interpretations differ, the same probability formulas apply to both. The axioms and rules of probability provide a consistent framework for reasoning under uncertainty, regardless of whether probability is interpreted as frequency or degree of belief.

## Random Variables

In probability theory, we rarely assign probabilities directly to raw outcomes. Instead, we define a **random variable**, which is a variable whose value depends on the outcome of an uncertain process. A random variable does not necessarily mean the process is truly random. It simply means that, from our perspective, the value is unknown until the outcome is observed. Random variables can be:

- **Discrete**, meaning they can take only a finite or [countably infinite](https://en.wikipedia.org/wiki/Countable_set) set of values (e.g. $0,1,2,\dots$).
- **Continuous**, meaning they can take any real value in an interval (e.g. any number in $[0,1]$).

For example, the result of a coin toss can be modeled as a discrete random variable $X \in \{0,1\}$, while the temperature measured by a sensor is naturally modeled as a continuous random variable. In probability notation, we usually write a random variable using a capital letter such as $X$, and a specific realized value using a lowercase letter such as $x$.

!!! note
    In deep learning, we often model data as random variables. For example, an image can be treated as a random variable $X$, and its label (such as "cat" or "dog") as another random variable $Y$. The goal of learning is then to discover patterns in how these random variables relate to each other.

## Probability Distributions

A random variable by itself only describes what values are possible. To reason quantitatively, we must also specify a **probability distribution**, which assigns probabilities to the different values the random variable can take.

Once the outcome is observed, the random variable takes a specific value. If a random variable is discrete, its distribution is described by a **probability mass function (PMF)**, denoted by $P(X=x)$. It assigns a probability to each possible value, such that:

- $0 \le P(X=x) \le 1$
- $\sum_x P(X=x) = 1$

If a random variable is continuous, its distribution is described by a **probability density function (PDF)**, denoted by $p(x)$. The PDF must satisfy:

- $p(x) \ge 0$
- $\int_{-\infty}^{\infty} p(x)\,dx = 1$

!!! warning "Important"
    For continuous variables, $p(x)$ is a _density_, not a probability. The value $p(x)$ can be greater than $1$. Probabilities are obtained only by integrating over an interval:
    $$
    P(a \le X \le b) = \int_a^b p(x)\,dx
    $$
    The probability of observing any exact value $X=x$ is always $0$. For example, if $X$ represents a real-valued measurement such as temperature, the probability of observing exactly $20.000^\circ$ is essentially zero, because the measurement could always end up being $19.999$ or $20.001$ instead. Only intervals have non-zero probability, such as $P(19.9 \le X \le 20.1)$.

!!! tip
    For integration and related topics, see the page dedicated to deep learning [Calculus](../01_calculus).


## Joint and Marginal Distributions

So far, we have described probability distributions over a single random variable. In many real problems, we must model multiple random variables at the same time. The probability distribution over two variables $X$ and $Y$ together is called the **joint distribution**. If $X$ and $Y$ are discrete, the joint distribution is written as:
$P(X=x, Y=y).$
If $X$ and $Y$ are continuous, the joint distribution is written as a joint density:
$p(x,y).$

Often, we are only interested in the distribution of one variable by itself. This is called the **marginal distribution**, and it can be obtained by summing (discrete case) or integrating (continuous case) over the other variable. For discrete variables:
$$
P(X=x) = \sum_y P(X=x, Y=y).
$$
For continuous variables:
$$
p(x) = \int p(x,y)\,dy.
$$

!!! note
    For example, suppose $X$ represents the outcome of a coin toss (tails or heads), and $Y$ represents the number shown on a fair die ($1$ to $6$). If we assume the coin toss does not affect the die roll (and vice versa), then each of the $2 \times 6 = 12$ outcomes is equally likely, so the joint distribution assigns probability $1/12$ to every possible pair:

    | $X \backslash Y$ | $1$  | $2$  | $3$  | $4$  | $5$  | $6$  |
    |------------------|------|------|------|------|------|------|
    | tails            | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 |
    | heads            | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 |

    To compute the marginal distribution of $X$, we sum across each row:[^marginal]
    $$
    P(X=\text{tails}) = \sum_{y=1}^{6} P(X=\text{tails},Y=y) = \frac{1}{2}.
    $$

    !!! success "Exercise"
        Compute the marginal distributions $P(X=\text{heads})$ and $P(Y=1)$.

## Conditional Probability

In many situations, we are interested in the probability of an event given that another event has already occurred. This is called **conditional probability**. The conditional probability of $A$ given $B$ is denoted by $P(A \mid B)$.

For discrete random variables $X$ and $Y$, the conditional distribution of $Y$ given $X$ is defined as:
$$
P(Y=y \mid X=x) = \frac{P(X=x, Y=y)}{P(X=x)}.
$$

!!! warning "Important"
    Many textbooks use shorthand notation such as $P(y \mid x)$ instead of $P(Y=y \mid X=x)$. We will mostly use explicit notation for clarity.

For continuous random variables, we use probability densities instead:
$$
p(y \mid x) = \frac{p(x,y)}{p(x)}.
$$

These formulas are only defined when $P(X=x)>0$ or $p(x)>0$, since we cannot condition on an event that never occurs.

!!! note
    Suppose $X$ is an image and $Y$ is its label. The joint distribution $P(X,Y)$ describes how often we encounter a specific image together with its correct label. The marginal distribution $P(X)$ describes what kinds of images appear in the world or in our dataset, regardless of their labels. The conditional distribution $P(Y \mid X)$ describes the probability of each label given a particular image. When we train a classifier, we are essentially training a model that takes an image $x$ and outputs estimates of probabilities like $P(Y=\text{cat} \mid X=x)$ and $P(Y=\text{dog} \mid X=x)$, where $x$ could be a particular input image provided to our model.

## Independence

In many problems, we work with multiple random variables. The relationship between these variables determines how complicated the joint distribution is. Two random variables $X$ and $Y$ are called **independent** if knowing the value of one gives no information about the other. Formally, $X$ and $Y$ are independent if their joint distribution factorizes into a product of marginals:
$$
P(X=x, Y=y) = P(X=x)\,P(Y=y).
$$

Equivalently, independence can be expressed using conditional probability:
$$
P(Y=y \mid X=x) = P(Y=y),
$$
meaning that observing $X$ does not change the probability of $Y$. For continuous variables, the same definition applies using probability densities:
$$
p(x,y) = p(x)\,p(y).
$$

!!! note
    Independence is a very strong assumption. In real-world data, variables are often correlated. However, independence assumptions are extremely useful because they allow us to build computationally efficient models.

Sometimes variables are not independent in general, but become independent once we condition on a third variable. We say that $X$ and $Y$ are **conditionally independent** given $Z$ if:
$$
P(X=x, Y=y \mid Z=z) = \\\\ 
P(X=x \mid Z=z)\,P(Y=y \mid Z=z).
$$

!!! note
    Suppose $Z$ represents whether it is raining. Let $X$ be whether the street is wet, and $Y$ be whether people are carrying umbrellas.  
    
    In general, $X$ and $Y$ are strongly correlated: if the street is wet, umbrellas are more likely. However, once we condition on $Z$ (rain), the relationship mostly disappears: given that it is raining, the street being wet does not provide much additional information about umbrellas. This illustrates conditional independence: $X \perp Y \mid Z$.

## Chain Rule

Probability has its own chain rule. Even when random variables are not independent, we can still represent any joint distribution by repeatedly applying the definition of conditional probability. For two variables, the joint distribution can be rewritten as:
$$
P(X=x, Y=y) = P(Y=y \mid X=x)\,P(X=x).
$$

In general, for $n$ random variables $(X_1, X_2, \dots, X_n)$, we can expand the joint distribution as:
$$
P(X_1, X_2, \dots, X_n)
=
\prod_{i=1}^{n} P(X_i \mid X_1, \dots, X_{i-1}).
$$

This identity is called the **chain rule** of probability. It is simply a consequence of how conditional probability is defined.

The chain rule gives a valid factorization, but it is often too complex because each conditional distribution depends on many variables. Independence assumptions simplify the factorization. For example, if we assume $X$ and $Y$ are independent, then:
$$
P(X,Y) = P(X)\,P(Y).
$$

If we assume $X$ and $Y$ are conditionally independent given $Z$, then:
$$
P(X,Y,Z) = P(X \mid Z)\,P(Y \mid Z)\,P(Z).
$$

This type of factorization is the foundation of many probabilistic models, including Bayesian networks and graphical models.

## Independent and Identically Distributed

In deep learning, the most common simplifying assumption is that training examples are **independent and identically distributed (i.i.d.)**. Suppose we have a dataset of samples:
$$
\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}.
$$

The i.i.d. assumption means that each sample is generated independently of the others, and all samples come from the same underlying distribution. Formally, if each sample is drawn from the same distribution $P(X)$ and samples are independent, then the joint probability of the dataset factorizes as:
$$
P(x^{(1)}, x^{(2)}, \dots, x^{(m)})
=
\prod_{i=1}^{m} P(x^{(i)}).
$$

This assumption is extremely important because it makes learning feasible: it allows the likelihood of a dataset to be written as a product, and the log-likelihood as a sum.

!!! warning "Important"
    The i.i.d. assumption is often violated in practice. Examples include time series, video frames, financial data, and datasets affected by distribution shift.   However, i.i.d. is still widely used because it provides a simple baseline model of how data is generated.

!!! note
    [Stochastic gradient descent (SGD)](../../notebooks/04_regul_optim) implicitly relies on the i.i.d. assumption: each mini-batch is treated as a random sample from the same distribution, so its gradient is assumed to approximate the full dataset gradient.

## Bayes' Rule

We have reached a crucial point in probability theory. Terminology can be dense and seem complicated, so I suggest spending some time here to clearly understand the concepts. You will frequently see the described terminology in deep learning literature. 

!!! tip
    For a visual intuition of Bayes' rule, see the video below on Bayes' theorem.

<div style="display:flex;justify-content:center;margin:1rem 0;">
  <div style="width:80%;max-width:900px;position:relative;padding-bottom:56.25%;height:0;overflow:hidden;">
    <iframe
      src="https://www.youtube.com/embed/HZGCoVF3YvM"
      style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
      allowfullscreen>
    </iframe>
  </div>
</div>


[Bayes’ rule](https://en.wikipedia.org/wiki/Bayes%27_theorem) allows us to reverse conditional probabilities. It provides a way to compute the probability of a hypothesis after observing **evidence**. For discrete random variables, Bayes’ rule is:
$$
P(X=x \mid Y=y) = \frac{P(Y=y \mid X=x)\,P(X=x)}{P(Y=y)}.
$$

Here, $P(X=x)$ is called the **prior**. It describes how likely $x$ was before observing $y$. The term $P(Y=y \mid X=x)$ is the **likelihood**. It measures how compatible the observation $y$ is with the hypothesis $x$. The result $P(X=x \mid Y=y)$ is called the **posterior**.

!!! notes
    The denominator $P(Y=y)$ acts as a normalization constant. It ensures that the posterior distribution $P(X \mid Y=y)$ sums to $1$ over all possible values of $X$.  

For continuous random variables, we use probability densities instead:
$$
p(x \mid y) = \frac{p(y \mid x)\,p(x)}{p(y)}.
$$

The denominator $p(y)$ is the marginal probability density of observing $y$. It is obtained by summing over all possible values of $x$ that could have produced $y$ (in the continuous case, summation becomes integration):
$$
p(y) = \int p(y \mid x)\,p(x)\,dx.
$$


!!! note
    This continuous form of Bayes' rule will become important for us later when discussing [variational autoencoders](../../notebooks/07_vae).

!!! note
    Consider a spam detection example. Let $X$ represent whether an email is spam ($X=\text{spam}$ or $X=\text{not spam}$), and let $Y$ represent whether the email contains the phrase "win money" ($Y=\text{yes}$ or $Y=\text{no}$). 
    
    Suppose only $2\%$ of all emails are spam, so the prior probability is $P(\text{spam})=0.02$ and $P(\text{not spam})=0.98$. Now assume spam emails contain the phrase "win money" $60\%$ of the time, so $P(\text{yes} \mid \text{spam})=0.60$, while normal emails contain it only $1\%$ of the time, so $P(\text{yes} \mid \text{not spam})=0.01$. If we observe an email containing "win money", Bayes’ rule gives:
    $$
    P(\text{spam} \mid \text{yes})
    =
    \frac{P(\text{yes} \mid \text{spam})P(\text{spam})}
    {P(\text{yes})}.
    $$
    The numerator is $0.60 \cdot 0.02 = 0.012$. The denominator is computed by marginalization:
    $$
    P(\text{yes})
    =
    P(\text{yes} \mid \text{spam})P(\text{spam})
    + \\\\
    P(\text{yes} \mid \text{not spam})P(\text{not spam})
    = \\\\
    0.60\cdot 0.02 + 0.01\cdot 0.98
    =
    0.0218.
    $$
    Therefore,
    $$
    P(\text{spam} \mid \text{yes}) = \frac{0.012}{0.0218} \approx 0.55.
    $$
    After observing the phrase, the probability that the email is spam jumps from $2\%$ to about $55\%$. This illustrates Bayes’ rule as a mechanism for updating beliefs: the likelihood tells us how strongly the evidence points toward spam, while the prior reflects how common spam is overall.








[^kolmogorov]: Kolmogorov, A.N. (1933, 1950). [Foundations of the theory of probability](https://archive.org/details/foundationsofthe00kolm). New York, US: Chelsea Publishing Company.

[^dutchbook]: A [Dutch book argument](https://en.wikipedia.org/wiki/Dutch_book_theorems) says that if your probability assignments are inconsistent, someone can design a set of bets that guarantees you lose money no matter what happens. For example, if you assign $P(A)=0.6$ and $P(\neg A)=0.6$, then you are claiming both an event and its opposite are more likely than not. A bettor could sell you a bet on $A$ and also sell you a bet on $\neg A$, and you would overpay in total, while only one of them can ever pay out. This guarantees a loss. The probability axioms prevent such contradictions.

[^frequentist]: A real coin is not infinitely thin, so in principle it could land on its edge. In practice this outcome is extremely rare, so it is usually ignored and the sample space is approximated as having only two outcomes.

[^marginal]: The term *marginal* is said to come from the traditional way of computing these sums on paper: one writes the joint distribution in a table and records the row and column totals in the margins of the page.



