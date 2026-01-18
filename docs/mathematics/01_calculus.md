# Calculus

DL is built on the idea that a model can be improved by making many small changes to its parameters. Calculus is the mathematical language that describes how small changes in inputs lead to changes in outputs. In DL, calculus is not used for symbolic manipulation or long derivations. It is used to measure sensitivity, guide optimization, and propagate information through a model.

This text explains the parts of calculus that matter for DL and why they matter.

## Functions and change

A function maps inputs to outputs. If we write

$$
y = f(x),
$$

then changing the input \(x\) usually changes the output \(y\). The central question of calculus is: how does the output change when the input changes?

If we change the input from \(x\) to \(x+h\), where \(h\) is a small number, the change in the output is

$$
f(x+h) - f(x).
$$

The ratio

$$
\frac{f(x+h) - f(x)}{h}
$$

describes the average rate of change of the function over this small step.

## Derivative: local rate of change

The derivative is defined as the limit of this ratio as the step size becomes very small:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.
$$

In simple words, the derivative tells us how sensitive the output is to a tiny change in the input. It answers the question: if I slightly increase \(x\), will the output increase or decrease, and by how much?

The most important interpretation for DL is local linear approximation. Near a point \(x\), a smooth function behaves approximately like a straight line:

$$
f(x+h) \approx f(x) + f'(x)\,h.
$$

This approximation explains why derivatives are useful for optimization. If we know the derivative, we can predict how the output will change for a small update and choose updates that reduce error.

## Derivatives of simple operations

Neural networks are complicated, but they are built from very simple operations. Calculus works because we can understand how each simple operation behaves and then combine these behaviors.

### Addition

Consider the function

$$
f(x) = x + c,
$$

where \(c\) is a constant. If we increase \(x\) by a small amount \(h\), the output increases by the same amount. The derivative is

$$
\frac{d}{dx}(x+c) = 1.
$$

If we define a function of two variables,

$$
z = x + y,
$$

then changing \(x\) by a small amount changes \(z\) by the same amount, and the same is true for \(y\). The sensitivity of \(z\) to each input is constant and equal to one.

### Multiplication

Now consider

$$
f(x) = c x,
$$

where \(c\) is a constant. Increasing \(x\) by a small amount increases the output by \(c\) times that amount. The derivative is

$$
\frac{d}{dx}(cx) = c.
$$

For two variables,

$$
z = x y,
$$

the sensitivity depends on the other variable. If we change \(x\) while keeping \(y\) fixed, the output changes proportionally to \(y\). If we change \(y\), the output changes proportionally to \(x\). This gives

$$
\frac{\partial z}{\partial x} = y, \qquad
\frac{\partial z}{\partial y} = x.
$$

This simple rule appears constantly in DL.

### A more complex example

Consider the function

$$
f(a,b,c) = a^2 + b^3 - c.
$$

If we change \(a\), only the \(a^2\) term is affected. The sensitivity of the output to \(a\) increases with the size of \(a\):

$$
\frac{\partial f}{\partial a} = 2a.
$$

If we change \(b\), the sensitivity depends on \(b^2\):

$$
\frac{\partial f}{\partial b} = 3b^2.
$$

If we change \(c\), the output decreases by exactly the same amount:

$$
\frac{\partial f}{\partial c} = -1.
$$

Each partial derivative measures how the output responds to changes in one variable, with all other variables held fixed.

## Partial derivatives and many parameters

DL models depend on many parameters at the same time. If the loss is written as

$$
L = f(\theta_1, \theta_2, \dots, \theta_n),
$$

then there is one partial derivative for each parameter. The partial derivative

$$
\frac{\partial L}{\partial \theta_i}
$$

measures how sensitive the loss is to parameter \(\theta_i\) alone.

In practice, partial derivatives are often approximated numerically using small changes, but training relies on exact derivatives computed efficiently by automatic differentiation.

## Gradient: collecting sensitivities

The gradient collects all partial derivatives into a single object:

$$
\nabla_{\theta} L =
\left[
\frac{\partial L}{\partial \theta_1},
\frac{\partial L}{\partial \theta_2},
\dots,
\frac{\partial L}{\partial \theta_n}
\right].
$$

The gradient tells us which parameters matter most at the current point and in which direction the loss increases fastest. This information is enough to guide learning.

## Gradient descent and learning

Training a DL model usually means repeatedly updating parameters using the gradient:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L,
$$

where \(\eta\) is a small positive number called the learning rate.

This update rule works because of local linearity. For small updates, the loss changes approximately linearly, and moving in the opposite direction of the gradient tends to reduce it.

## The chain rule

Models in DL are compositions of many functions. If

$$
y = g(x), \qquad L = f(y),
$$

then the loss is a function of \(x\) through \(y\). The chain rule states

$$
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}.
$$

In words, the sensitivity of the loss to \(x\) is the sensitivity of the loss to \(y\), multiplied by the sensitivity of \(y\) to \(x\).

When many operations are chained together, the chain rule applies repeatedly. Backpropagation is the systematic application of the chain rule through a computation graph, starting from the loss and moving backward to the parameters.

## Smoothness and why gradients work

Gradient-based learning assumes that the loss changes smoothly with respect to the parameters. This means that small changes in parameters produce small and predictable changes in the loss. If this were not true, derivatives would not provide reliable guidance.

This assumption is usually satisfied in DL because models are built from smooth operations such as addition, multiplication, and smooth nonlinear functions.

## Integration and accumulation

Differentiation measures instantaneous change. Integration measures accumulated change. The integral

$$
\int_a^b f(x)\,dx
$$

represents the total accumulation of the quantity \(f(x)\) over an interval.

In DL, integration rarely appears as a calculation done by hand. Instead, it appears conceptually in expectations, averages over data, and probability distributions.

## The Fundamental Theorem of Calculus

The Fundamental Theorem of Calculus connects differentiation and integration. If

$$
F(x) = \int_a^x f(t)\,dt,
$$

then

$$
F'(x) = f(x).
$$

This means that accumulation and local change are two sides of the same idea. In DL, this connection underlies the relationship between local gradients and the overall effect of many updates over time.

## Convexity and modern deep learning

In classical optimization, convex functions are important because they guarantee a single global minimum. Most DL loss functions are not convex. Despite this, gradient-based methods work well in practice. Local gradient information is often sufficient to make steady progress, even when the global shape of the loss surface is complex.

## Summary

The calculus used in DL focuses on local change, sensitivity, and composition. Derivatives measure how outputs respond to small input changes. Partial derivatives isolate the effect of individual parameters. Gradients collect these effects into a usable form. The chain rule explains how sensitivities propagate through a model. Integration provides the idea of accumulation, even when no integrals are computed explicitly.

Together, these ideas form the calculus foundation of deep learning.
