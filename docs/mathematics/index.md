# Mathematics of Deep Learning

<div style="margin:.3rem 0 1rem;font-size:.9em;color:#555;display:flex;align-items:center;gap:.35rem;font-family:monospace">
  <time datetime="2026-01-18">18 Jan 2026</time> ·
  <time datetime="PT9M">9 min</time>
</div>

Deep Learning (DL) relies on mathematics, but not on all of mathematics equally. Many topics that are common in standard mathematics curricula play little or no role in the practice of DL. The purpose of this section is to explain which parts of mathematics matter for DL and what role they play. 

In preparing this material, two widely used resources were consulted and found to be highly valuable: [Deep Learning (Goodfellow et al., 2016)](https://www.deeplearningbook.org) and [Dive into Deep Learning (Zhang et al., online)](https://d2l.ai).

*Deep Learning* presents the mathematics in a concise and rigorous form. Its strength lies in precision and breadth, but this compact style can make it difficult for readers to develop intuition, especially when encountering these ideas for the first time. Key concepts are often introduced quickly, with limited space for informal explanation or gradual buildup.

*Dive into Deep Learning* takes a different approach, tightly integrating mathematical ideas with executable code. This makes experimentation accessible and practical, but it can also blur the boundary between mathematical concepts and their implementation. The mathematical knowledge is not always presented in a clearly systematized form.

The goal of the present material is to combine the strengths of both approaches while addressing their limitations. Mathematical ideas are introduced carefully and explained in simple language, with implementation details separated whenever possible. Each concept is included because it plays a clear role in DL, not because it belongs to a traditional mathematics curriculum. The aim is to provide a conceptual foundation that supports both practical experimentation and deeper theoretical study.

More detailed overviews can be found in the separate pages dedicated to [Calculus](01_calculus), [Linear Algebra](02_linear_algebra), [Probability Theory](03_probability), and [Information Theory](04_information). Below is a summary of the main mathematical concepts required for DL.

## Calculus

Within calculus, the central idea for DL is the rate of change: *if we change some model parameters slightly, how does the output change?* In DL, the output of interest is usually a single number called the *loss*, which measures how bad the model's prediction is. A **derivative** tells us how much the loss changes when we slightly change one parameter. This makes derivatives a practical tool for learning, since they indicate the direction in which parameters should be adjusted to reduce the loss.

**Partial derivatives** are essential because a model typically has many parameters. A partial derivative measures how sensitive the loss is to one parameter while all other parameters are kept fixed. The **gradient** simply collects all these sensitivities into a single array. The **chain rule** explains how sensitivities propagate through a model that is built from many smaller operations, and *backpropagation* is the algorithm that applies the chain rule efficiently to compute gradients.

All of this relies on an important assumption: the loss changes smoothly with respect to the parameters. This means that small changes in parameters lead to small, predictable changes in the loss, making derivatives reliable guides for *optimization*.

**Integration** and the *Fundamental Theorem of Calculus* appear more quietly in the background. They underlie concepts such as *expectations* and *averages*, which are central to training objectives. In DL, integrals are rarely computed by hand; _understanding what integration represents is more important than learning how to calculate it_.

## Linear Algebra

Linear algebra is the language in which DL models are written. Data points, parameters, and gradients are represented as *vectors*. Linear layers are represented as *matrix–vector* or *matrix–matrix multiplications*.

What matters most is intuition. **Vectors** should be understood as ordered collections of numbers. **Matrices** should be understood as operations that transform vectors by scaling, rotating, or mixing their components. These ideas explain how information flows through a network and why many computations can be done efficiently in parallel.

Linear algebra also explains why gradients have the same shape as parameters, why batching works, and why modern hardware such as GPUs is effective for DL. Model parameters are stored as vectors and matrices, and gradients are derivatives with respect to those parameters, so they naturally have the same structure. This one-to-one correspondence makes parameter updates straightforward: each parameter is adjusted using its matching gradient entry.

Modern DL frameworks are built almost entirely on linear algebra operations. Matrix multiplication, matrix addition, and vectorized nonlinear functions form the core of both the forward and backward passes. Most performance optimizations are handled automatically by numerical libraries, allowing users to express models at a high level while relying on efficient low-level implementations. 

While **matrix factorizations** such as [Singular Value Decomposition (SVD)](https://shahaliyev.org/writings/svd) are rarely invoked explicitly during training, they are used internally in optimized linear solvers, low-rank approximations, spectral normalization, and in estimating matrix norms or conditioning. Through these mechanisms, factorization-based ideas influence numerical stability, efficiency, and scaling behavior in DL systems without appearing directly in model code.


Batching works because linear algebra operations naturally extend from single vectors to collections of vectors stacked into matrices or higher-dimensional tensors. Processing many data points at once is not a special trick, but a direct consequence of writing models in matrix form. GPUs are effective for DL for the same reason: linear algebra operations consist of many simple arithmetic operations that can be carried out *in parallel*. As a result, DL benefits directly from both the mathematical structure of linear algebra and the hardware designed to execute it efficiently.

## Probability Theory

DL models are trained on data that is noisy, incomplete, and often ambiguous. Probability provides the language for describing this uncertainty and for turning learning into a well-defined mathematical problem. In DL, models are often best understood not as systems that produce a single "correct" output, but as systems that assign probabilities to possible outcomes.

From this perspective, a model defines a **probability distribution**, either explicitly or implicitly. Training the model means adjusting its parameters so that the observed data becomes more probable under this distribution. Many commonly used loss functions arise directly from this idea. Minimizing such losses is equivalent to **maximizing likelihood**.

**Expectations** play a central role because learning is not based on a single data point, but on averages over data drawn from an underlying distribution. Training objectives are typically expectations of a loss over the data distribution, which in practice are approximated using finite datasets and minibatches.

DL does not require advanced probability theory, but it does require a clear understanding of what **probabilistic models** represent, how likelihood and expectation relate to loss functions, and why uncertainty is an essential part of learning from real data.

## Information Theory

Information theory enters DL when we want to measure how different two probability distributions are. Many DL models define a distribution over possible outputs rather than producing a single fixed prediction. Information-theoretic quantities provide a principled way to compare these predicted distributions to the true data distribution.

Concepts such as **entropy** and **cross-entropy** arise naturally in this setting. Entropy measures uncertainty, while cross-entropy measures how well one distribution represents another. Minimizing cross-entropy encourages the model to assign high probability to the observed data.

A closely related quantity is the **Kullback–Leibler (KL) divergence**, which measures how much information is lost when one distribution is used to approximate another. Many common training objectives can be interpreted as minimizing a KL divergence, even when this connection is not stated explicitly.

## Additional Mathematics

In addition to the core areas discussed above, several mathematical perspectives play an important role in DL. While they may not always appear as standalone topics or require extensive formal development, they shape how models are designed, trained, and evaluated. These ideas recur across many DL settings and deserve explicit attention, even when they are introduced briefly.

**Statistics** enters DL through the fact that models are trained on finite samples rather than full data-generating processes. Concepts such as _generalization_, _overfitting_, and the _bias–variance tradeoff_ describe the structural limits of what can be learned from data and how model complexity interacts with sample size and noise. These considerations shape how results should be interpreted, how sensitive conclusions are to data variation, and how confidently performance can be expected to transfer beyond the observed sample.

**Optimization theory** addresses a small set of practical questions that arise once a loss function is defined. Given a highly _non-convex_ objective with millions of parameters, how can it be minimized efficiently, and why do simple gradient-based methods work at all? How do learning rates, momentum, adaptive updates, and noise from minibatching affect training behavior? Rather than providing exact convergence proofs, optimization theory offers guidance on training stability, speed, and failure modes.

**Geometry** treats representations, parameters, and activations as points in high-dimensional spaces. Distances and angles define similarity, with measures such as cosine similarity capturing directional alignment between representations. Optimization itself is a geometric process, moving parameters across a loss surface whose local curvature influences learning speed and stability. Geometric intuition about distances, neighborhoods, and curvature helps explain why certain architectures, losses, and similarity measures are effective in practice.

**Graph theory** becomes relevant whenever data is structured by relations rather than simple vectors. In graph neural networks (GNNs), data is represented as nodes and edges, and learning depends directly on graph connectivity. Related ideas also appear more broadly in message passing, relational reasoning, and attention mechanisms applied to structured inputs.

**Numerical computation and stability** constrain how DL models are implemented. Because training relies on finite-precision arithmetic, issues such as overflow, underflow, and loss of precision directly influence model behavior. Many standard techniques in DL—such as normalization layers, carefully designed loss functions, and specific activation choices—exist primarily to ensure stable and reliable computation.

Together, these perspectives complement the core mathematical foundations and connect them to practical modeling, training, and evaluation. They do not replace the core framework, but they shape how it is applied and understood in real-world deep learning systems.
