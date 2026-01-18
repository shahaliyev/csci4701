# Calculus

<div style="margin:.3rem 0 1rem;font-size:.9em;color:#555;display:flex;align-items:center;gap:.35rem;font-family:monospace">
  <time datetime="2026-01-18">18 Jan 2026</time> ·
  <time datetime="PT12M">12 min</time>
</div>

Calculus studies two closely related ideas: **accumulation** (integration) and **change** (differentiation). In DL, learning is defined by accumulating error across data, usually as an average loss. Training then proceeds by making small changes to model parameters in order to reduce this accumulated error. Calculus provides the language and structure for both. This section builds calculus concepts from fundamentals, with the goal of understanding how they support learning, optimization, and model behavior in deep learning.

## Functions

A **function** maps inputs to outputs. We write this as $y = f(x)$. If the input $x$ changes, the output $y$ usually changes as well. Some functions change slowly, some change quickly, and some change differently depending on at which point of the function you are. Calculus begins by asking how these changes are related. 

!!! note
    For example, changing the brightness of an image slightly may barely affect a model's output in one case, but cause a large change in another.


## Integration

An **integral** such as $\int_a^b f(x)\,dx$ represents the total accumulation of the values of $f(x)$ as $x$ moves from $a$ to $b$. It is simply the continuous analogue of a summation. You can think of an integral as summing many small contributions of $f(x)$ over an interval. The exact techniques for computing integrals are less important in DL than the idea they represent.

Conceptually, integration means breaking an interval into many small pieces. For each piece, we take the value of $f(x)$ and multiply it by the width of the piece. Adding all these pieces together gives an approximation of the total accumulation. As the pieces become smaller and more numerous, this approximation approaches the integral. Mathematically, we represent this very small width as $dx$.


<figure>
  <img src="../../assets/images/calculus/integration.gif" alt="Integration" style="max-width: 100%; height: auto;">
  <figcaption style="margin-top: 0.5em; font-size: 0.9em; opacity: 0.85;">
    Riemann Integration and Darboux Lower Sums. By <a href="//commons.wikimedia.org/wiki/User:IkamusumeFan" title="User:IkamusumeFan">IkamusumeFan</a> - <span class="int-own-work" lang="en">Own work</span><span typeof="mw:File"><a href="//commons.wikimedia.org/wiki/File:Matplotlib_icon.svg" class="mw-file-description"></a></span>&nbsp;This plot was created with <a href="https://en.wikipedia.org/wiki/en:Matplotlib" class="extiw" title="w:en:Matplotlib"><span title="comprehensive library for creating static, animated, and interactive visualizations in Python">Matplotlib</span></a>., <a href="https://creativecommons.org/licenses/by-sa/3.0" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=28296068">Link</a>
  </figcaption>
</figure>


!!! note
    In DL, training is never based on a single example. A model is evaluated by how it performs across many examples, so errors must be combined into one overall value. In practice, we only have access to a limited number of training examples. A common case in ML and DL is <a href='https://en.wikipedia.org/wiki/Mean_squared_error'>mean squared error (MSE)</a>, where for each training sample we compute a prediction error, square it (so negative and positive errors do not cancel, and larger mistakes are penalized), and then average these squared errors over the dataset. Conceptually, however, this dataset-level average is not the final goal. 

The dataset is usually treated as a small collection of examples drawn from a much larger source of data. Ideally, we would like to measure the model’s average error over all possible data points it might encounter, not just the ones we happened to collect. The finite average used in training should therefore be understood as a practical **approximation** of a more general, ideal accumulated continuous quantity. For the values $g(x_1), g(x_2), \dots, g(x_N)$, we can write

$$
\mathbb{E}_{x \sim p}[g(x)] \approx \frac{1}{N}\sum_{i=1}^N g(x_i).
$$

Here, the right-hand side is what we compute from data, and the left-hand side represents the ideal quantity we are trying to approximate. This ideal accumulated quantity is written precisely using the concept of an **expectation**. When data is described by a probability distribution $p(x)$, the average value of a quantity $g(x)$ is written as

$$
\mathbb{E}_{x \sim p}[g(x)] = \int g(x)\,p(x)\,dx.
$$

You do not need a deep understanding of probability to read this expression. Conceptually, it means: consider all possible values of $x$, weight each value by how common it is, and add everything up. Many DL loss functions can be understood this way, as average losses over all possible data. For discrete datasets, this expectation reduces to a finite sum, while for continuous variables it is written as an integral. The integral itself is not special—it is simply the mathematical way to express an average over all possible inputs when the space of inputs is continuous.

!!! tip
    If the idea of expectations or probability distributions feels unfamiliar, you may want to read the page dedicated to the
    <a href="../03_probability">Probability Theory</a> alongside this section.

## Differentiation

Differentiation answers the question: _if we change the input slightly, how much does the output change?_  Suppose we start at $x$ and then move a small step $h$ to $x+h$. The corresponding change in the output is $f(x+h) - f(x)$. By itself, this number depends on how large $h$ is. To describe change in a way that does not depend on the step size, we compare the output change to the input change by forming the ratio

$$
\frac{f(x+h) - f(x)}{h}.
$$

This ratio is called a _difference quotient_. It describes the **average rate of change** of the function over the small interval from $x$ to $x+h$. The **derivative** is defined as the limit of this ratio as the step size approaches zero:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.
$$

This definition captures the idea of an "instantaneous" rate of change. Intuitively, the derivative tells us the **slope** of the function at the point $x$: if $f'(x)$ is large, a tiny change in $x$ causes a large change in $f(x)$, if $f'(x)$ is close to zero, the function is locally flat. 


!!! note
    In DL, if increasing a model weight slightly increases the loss, then the derivative of the loss with respect to that weight is positive. That means decreasing the weight slightly should reduce the loss (at least locally).

The derivative is not only a number; it also provides a practical approximation of how a function behaves near a given point. The key idea is that, over very small distances, a smooth function behaves almost like a straight line. If we start at a point $x$ and move a small step $h$, the derivative $f'(x)$ tells us how steep the function is at $x$. Using this slope, we can estimate how much the output will change. This leads to the approximation

$$
f(x+h) \approx f(x) + f'(x)\,h.
$$

This formula should be read as a prediction: "start from the current value $f(x)$, then add the change suggested by the slope times the step size." The approximation becomes more accurate as the step $h$ becomes smaller. Geometrically, this means that near the point $x$, the function can be replaced by its tangent line. The tangent line touches the function at $x$ and has the same slope there. Over a very small region, the curve and the tangent line are almost indistinguishable, which is why the [linear approximation](https://en.wikipedia.org/wiki/Linear_approximation) works.

<figure>
  <img src="../../assets/images/calculus/tangent.svg" alt="Tangent line" style="max-width: 80%; height: auto;">
  <figcaption style="margin-top: 0.5em; font-size: 0.9em; opacity: 0.85;">
    By Chorch - Own Work, Public Domain, <a href="https://commons.wikimedia.org/w/index.php?curid=926971">Link</a>
  </figcaption>
</figure>

!!! note
    In DL, training works because, at each step, we treat the loss as locally almost linear in the parameters. The gradient (see below) gives the slope of this local linear approximation. By making small parameter updates in the direction opposite to the gradient, we can reliably reduce the loss step by step, even when the overall loss function is highly complex.


## Partial derivatives

Deep learning models depend on many parameters at once. If the loss is written as

$$
L = f(\theta_1, \theta_2, \dots, \theta_n),
$$

then each parameter has its own **partial derivative** $\frac{\partial L}{\partial \theta_i}.$ A partial derivative measures how the loss changes when one parameter is varied while all others are held fixed.

!!! note
    For example, changing a single weight in a neural network affects the loss while all other weights remain unchanged.
 
## Gradients

The **gradient** collects all partial derivatives into a single vector:

$$
\nabla_{\theta} L =
\left[
\frac{\partial L}{\partial \theta_1},
\frac{\partial L}{\partial \theta_2},
\dots,
\frac{\partial L}{\partial \theta_n}
\right].
$$

The gradient points in the direction where the loss increases most rapidly. Moving in the opposite direction locally reduces the loss. Each component of the gradient corresponds to one parameter and tells us how that parameter influences the loss. Training typically consists of repeated updates of the form

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L,
$$

where $\eta$ is the learning rate. Each update makes a small change. Over many updates, these small changes accumulate and reduce the overall loss.

!!! tip
    The learning rate update through backward pass is discussed in the notebook dedicated to [backpropagation](../notebooks/01_backprop.ipynb).

## Jacobian

The [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) is the general first-order derivative for functions with vector inputs and vector outputs. If a function maps an $n$-dimensional input vector to an $m$-dimensional output vector, $f : \mathbb{R}^n \rightarrow \mathbb{R}^m,$ its Jacobian is an $m \times n$ matrix defined as

$$
J =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \dots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \dots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \dots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}.
$$

Each entry measures how one component of the output changes when one component of the input is varied. The Jacobian therefore captures all first-order sensitivities between inputs and outputs.

!!! note
    In DL, layers often map vectors to vectors. Although Jacobians are rarely written explicitly, they are the objects through which changes propagate from one layer to the next. When the output is a scalar loss, the Jacobian reduces to a row vector. Conceptually, the gradient introduced earlier is simply the Jacobian of a scalar-valued function. Backpropagation avoids forming full Jacobian matrices explicitly. Instead, it efficiently computes vector–Jacobian products, which is why gradients can be computed for models with millions of parameters at reasonable cost.

!!! tip
    Jacobians are best understood through linear algebra. If matrices and vector transformations feel unfamiliar, you may want to read the <a href="../02_linear_algebra">Linear Algebra</a> page alongside this section.
    

## Chain Rule

DL models are built by *composing functions*. Instead of a single operation, a model applies many transformations one after another. Each transformation takes the output of the previous one as its input. To understand how changes propagate through such a model, consider a simple composition:

$$
y = g(x), \qquad L = f(y).
$$

Here, $x$ influences $L$ indirectly, through the intermediate variable $y$. If we change $x$ slightly, $y$ will change, and that change in $y$ will in turn affect $L$. The chain rule formalizes this dependency.

The [chain rule](https://en.wikipedia.org/wiki/Chain_rule) states that the sensitivity of $L$ with respect to $x$ is the product of two sensitivities:

$$
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}.
$$

This equation should be read step by step. First, $\frac{dy}{dx}$ tells us how a small change in $x$ affects $y$. Second, $\frac{dL}{dy}$ tells us how a small change in $y$ affects the loss. Multiplying them gives the total effect of changing $x$ on $L$.

This idea extends naturally to longer chains. If a model applies many functions in sequence, the chain rule is applied repeatedly, multiplying together the local sensitivities at each step. Each operation contributes a small piece to the overall gradient.

## Taylor Expansion

[Taylor series](https://en.wikipedia.org/wiki/Taylor_series) provides a systematic way to describe how a function behaves near a given point. It expresses a function as a sum of terms built from its derivatives at that point. Each term captures progressively finer details of how the function changes.

For a function $f(x)$ expanded around a point $x$, the Taylor series in one dimension is

$$
f(x+h) = f(x) + f'(x)h + \tfrac{1}{2}f''(x)h^2 + \tfrac{1}{6}f'''(x)h^3 + \dots
$$

This expression says that the value of the function at $x+h$ can be predicted by starting from the value at $x$ and then adding corrections based on information about how the function changes at $x$.

In practice, we rarely use the full infinite series. Instead, we keep only the first few terms. This truncated version is called a **Taylor expansion** and is used as a local approximation.

Keeping only the first-order term gives the linear approximation already used in gradient-based learning:

$$
f(x+h) \approx f(x) + f'(x)h.
$$

This approximation assumes that, for small updates, the function behaves almost like a straight line near the current point. It explains why gradients provide useful guidance for optimization.

This local linear approximation relies on an important assumption: the function must be **smooth enough** near the point of expansion. Smoothness means that small changes in the input lead to small, predictable changes in the output, and that derivatives do not change abruptly.

!!! note
    In DL, loss functions are often not perfectly smooth everywhere, but they are typically **piecewise smooth**. This is sufficient. Taylor expansions and gradient-based updates only rely on local behavior along the training trajectory, not on global smoothness of the loss surface. A common example is the [ReLU activation](../notebooks/02_neural_network.ipynb), which is not differentiable at zero but is differentiable almost everywhere else. Gradient-based methods rely on this local behavior and use subgradients at nondifferentiable points.

Keeping second-order terms reveals that this linear behavior is only approximate. These higher-order terms explain why the slope itself can change as we move, motivating the need to understand second-order structure.

!!! note
    In DL, gradient-based learning relies on first-order Taylor approximations. Understanding why and when this approximation breaks down requires looking at second-order effects, which are captured by the Hessian.

## Hessian

While the Jacobian describes first-order behavior—how the loss changes under small parameter changes—the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) describes second-order behavior.[^1] It captures how these first-order sensitivities themselves change as we move in parameter space. The Hessian of $L$ with respect to the parameter vector $\theta$ is a matrix of second-order partial derivatives:

$$
H =
\begin{bmatrix}
\frac{\partial^2 L}{\partial \theta_1^2} &
\frac{\partial^2 L}{\partial \theta_1 \partial \theta_2} &
\dots &
\frac{\partial^2 L}{\partial \theta_1 \partial \theta_n} \\[0.5em]
\frac{\partial^2 L}{\partial \theta_2 \partial \theta_1} &
\frac{\partial^2 L}{\partial \theta_2^2} &
\dots &
\frac{\partial^2 L}{\partial \theta_2 \partial \theta_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 L}{\partial \theta_n \partial \theta_1} &
\frac{\partial^2 L}{\partial \theta_n \partial \theta_2} &
\dots &
\frac{\partial^2 L}{\partial \theta_n^2}
\end{bmatrix}.
$$

Each entry tells us how the sensitivity with respect to one parameter changes when another parameter is varied. In this sense, the Hessian measures **curvature**: how the loss surface bends in different directions. Consider a simple two-parameter loss $L(\theta_1, \theta_2)$. The diagonal entries of the Hessian describe how sharply the loss curves when we move along each parameter direction individually. The off-diagonal entries describe how changes in one parameter affect the sensitivity with respect to another parameter.

!!! note
    In DL, this information explains important optimization behavior. Directions with strong positive curvature correspond to narrow valleys, where large updates can easily overshoot. Directions with weak curvature correspond to flat regions, where progress can be slow. Negative curvature indicates directions where the loss bends downward, which is typical near saddle points. Although full Hessians are rarely computed explicitly in DL due to their size and cost, their effects are always present. Learning rate selection, optimization stability, and the behavior of training near minima and saddle points are all influenced by second-order structure.

!!! tip
    Like the Jacobian, the Hessian is a linear algebra object—a matrix encoding directional behavior. If matrices, eigenvalues, or curvature interpretations feel unfamiliar, you may want to read the
    <a href="../02_linear_algebra">Linear Algebra</a> page alongside this section.


## Minima, saddle points, and convexity

A **minimum** is a point where small changes in any direction increase the loss. At such a point, the gradient is zero and the surrounding curvature points upward.

A **saddle point** is also a point where the gradient is zero, but the behavior is mixed: the loss increases in some directions and decreases in others. This means the point is neither a true minimum nor a maximum. The distinction between minima and saddle points is determined by the local curvature described by the Hessian.

!!! note
    In high-dimensional DL models, saddle points are far more common than poor local minima. Gradient-based methods can often escape saddle points because curvature creates unstable directions, and stochastic noise from minibatches helps push parameters away from them.

In classical optimization, **convex** loss functions play a special role. For a convex function, any point where the gradient is zero is guaranteed to be a global minimum. There are no saddle points and no spurious local minima.

!!! note
    Most DL loss functions are **not convex**. As a result, global guarantees do not apply. Instead, training relies on local information provided by gradients and curvature. Despite the lack of convexity, gradient-based methods work well in practice due to overparameterization, stochastic gradients, and the structure induced by modern architectures, even though no global guarantees apply.

Gradient-based learning does not require global convexity. What matters is that, locally, the loss behaves smoothly enough for gradients and Taylor approximations to provide reliable guidance along the training trajectory.

## Fundamental Theorem of Calculus

The [Fundamental Theorem of Calculus](https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus) explains the precise relationship between accumulation and change. If we define an accumulated quantity

$$
F(x) = \int_a^x f(t)\,dt,
$$

then $F(x)$ is differentiable and

$$
\frac{d}{dx} F(x) = f(x).
$$

This means that differentiation recovers the rate at which accumulation occurs. Conversely, if $F(x)$ is any antiderivative of $f(x)$, then total accumulation over an interval can be computed as

$$
\int_a^b f(x)\,dx = F(b) - F(a).
$$

Together, these statements show that local change and total accumulation are two sides of the same idea.

!!! note
    In DL, losses are defined as accumulated quantities, while gradients describe local change. Training works because following local gradients causes a consistent reduction in the accumulated loss over time.

[^1]: For a further DL–oriented treatment of gradients, Jacobians, Hessians, and numerical aspects of optimization, see Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. [Chapter 4: Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html).