# Linear Algebra

!!! warning "Important"
    The page is currently under development.

<div style="margin:.3rem 0 1rem;font-size:.9em;color:#555;display:flex;align-items:center;gap:.35rem;font-family:monospace">
  <time datetime="2026-01-19">19 Jan 2026</time> ·
  <!-- <time datetime="PT22M">22 min</time> -->
</div>

Linear algebra is the branch of mathematics that studies vector spaces and the linear mappings between them. In deep learning, almost all computation is formulated in the language of linear algebra: data, model parameters, activations, gradients are represented as vectors or matrices. A clear understanding of what these objects represent—and how they behave under linear operations—is necessary not only for correct implementation, but for reasoning about model structure, learning dynamics, and numerical behavior.


## Scalars and Vectors

A [scalar](https://en.wikipedia.org/wiki/Scalar_(mathematics)) is a single number (often real-valued). It is more challenging to define a **vector**. Different fields use the same object with different mental models and all are legitimate because they rely on the same linear rules.

From a mathematician's point of view, a vector is an element of a [vector space](https://en.wikipedia.org/wiki/Vector_space): something you can **add and scale while satisfying certain axioms** (closure, associativity, distributivity, etc.). The axioms exist to guarantee that linear combinations behave predictably. From this requirement of [linearity](https://en.wikipedia.org/wiki/Linearity) through addition and scalar scaling (homogeinity), you get a powerful combined statement:
$$
f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)
$$

!!! note
    Raw audio signals satisfy the linearity properties to a very good approximation. If two sounds are played at the same time, the resulting waveform is (approximately) the sum of the individual waveforms. If the volume of a sound is increased or decreased, its waveform is scaled by a constant factor. Because audio combines by superposition and scales linearly with amplitude, it can be naturally represented as a vector and manipulated using linear algebra.


From a physicist's point of view, a vector represents a **quantity with direction and magnitude** (velocity, force). You add forces, scale forces, decompose into components. The vector predicts physical behavior. From a computer scientist's point of view, a vector is simply an **array of numbers**. It can represent pixel values of an image, coordinates of a point, words in a document, etc.

!!! warning "Important"
     In many ways, machine/deep learning borrows terminology from mathematics and uses it rather freely. Terms like **vector**, **dimension**, **space**, **metric**, **manifold**, and even **linear** are frequently misused. For example, by "dimension" one could assume "tensor axis length". This is convenient shorthand, but it can break intuition if you don't keep in mind the underlying differences between deep learning and mathemtics which often use the same tools for different purposes.

A vector is often written explicitly as a column of numbers. For example, a vector with \(n\) real-valued components can be written as
$$
\mathbf{v} =
\begin{bmatrix}
v_1 \\\\
v_2 \\\\
\vdots \\\\
v_n
\end{bmatrix}
\in \mathbb{R}^n
$$

In deep learning, such a vector is typically understood operationally: it is stored as a contiguous array of \(n\) real numbers and is mathematically an element of \(\mathbb{R}^n\), the Cartesian product of \(\mathbb{R}\) with itself \(n\) times. In this context, its "dimension" refers simply to its length \(n\). When \(n = 2\) or \(n = 3\), the vector can be visualized geometrically as a point or an arrow. When \(n\) is large, direct visualization is no longer possible, but the same algebraic operations—addition, scalar multiplication, dot products, and linear transformations—still apply. 

!!! note
    Depending on context, vectors can be visualized in different ways. In geometry and physics, they are often drawn as arrows representing magnitude and direction. In other settings, a vector can be viewed as a function that assigns a value to each index or coordinate. These visualizations are useful for building intuition, especially in low dimensions, but they do not alter the underlying algebraic definition of a vector. Linear algebra itself does not rely on geometric interpretation. It is fundamentally an algebraic theory of vector spaces and linear maps, and all definitions and results are independent of visualization. Geometry serves only as an intuitive aid not as a prerequisite. Beyond three dimensions, geometry in the literal, visual sense becomes unusable. Since most representations in deep learning live in very high-dimensional spaces, geometric visualization is generally not available and plays no direct role in practice. What remains meaningful are algebraic and analytical notions—such as inner products, norms, projections, and linear maps—rather than pictures or spatial intuition.

## Matrices and Tensors

A [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)) is a rectangular array of numbers arranged in rows and columns, satisfying properties of addition and multiplication (perhaps, from both deep learning and mathematics points of view). Formally, a real-valued matrix with \(m\) rows and \(n\) columns is an element of \(\mathbb{R}^{m \times n}\):
$$
\mathbf{A} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\\\
a_{21} & a_{22} & \cdots & a_{2n} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$



From a mathematical point of view, a matrix represents a [linear map](https://en.wikipedia.org/wiki/Linear_map) between vector spaces. Given a vector \(\mathbf{x} \in \mathbb{R}^n\), multiplication by a matrix \(\mathbf{A} \in \mathbb{R}^{m \times n}\) produces a new vector \(\mathbf{y} \in \mathbb{R}^m\): $$
\mathbf{A}\mathbf{x} = \mathbf{y}.$$ 

This operation encodes all **linear transformations**: rotations, scalings, projections, and combinations of these. The key idea is that matrices do not just store numbers; they describe how vectors are transformed. In deep learning, matrices appear everywhere. For example, in

- Model parameters (weights of fully connected layers)
- Batches of input data
- Linear layers of the form \( \mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b} \)
- [Jacobians and Hessians](../01_calculus) (implicitly, through [automatic differentiation](../../notebooks/01_backprop))

A matrix is stored as a 2D array in memory, but it should be understood as a single object representing a linear operation. Confusing these two viewpoints—matrix as data vs. matrix as transformation—is a common source of misunderstanding.

!!! note 
    When training neural networks, we rarely reason about individual entries of a matrix. Instead, we reason about the effect of the matrix as a whole: how it mixes input features, how it changes dimensionality, and how it interacts with nonlinearities. Frameworks exploit this by implementing matrix multiplication using highly optimized numerical kernels.

A [tensor](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) is a generalization of scalars, vectors, and matrices to higher dimensions. Informally: a scalar is a 0-order tensor, a matrix is a 2-order tensor, etc. In deep learning practice, a tensor is best understood as a multidimensional array of numbers with a fixed shape.

!!! warning "Important"
    In pure mathematics and physics, tensors have a precise coordinate-independent definition involving multilinear maps. In deep learning, the word *tensor* is used more loosely to mean _n_-dimensional array. This is a practical simplification, but it is important not to confuse it with the full mathematical theory of tensors.

Tensor operations in deep learning are designed to preserve linear structure wherever possible. Linear operations (e.g. matrix multiplication, [convolution](../../notebooks/03_cnn_torch)) remain linear even when expressed in tensor form. Most neural network layers can be viewed as linear maps acting on tensors, followed by nonlinear functions applied elementwise. Understanding which parts of a computation are linear and which are not is essential for reasoning about optimization and numerical stability.

!!! note
    High-dimensional tensors cannot be visualized geometrically. Their meaning comes from structure and indexing, not from geometrical intuition. What matters is how dimensions correspond to data and how linear operations act along specific axes (e.g. columns, rows).

## Transpose, Identity, Inversion

The **transpose** of a matrix swaps rows and columns. For a matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$, its transpose $\mathbf{A}^\top\in\mathbb{R}^{n\times m}$ is defined by $\mathbf{A}^\top_{ij} = a_{ji}.$ A column vector becomes a row vector, and vice versa. Basically, transpose reflects (like a mirror) a matrix across its main diagonal. Elements on the diagonal remain fixed, off-diagonal elements are mirrored:
$$
\mathbf{A} =
\left[
\begin{array}{ccc}
a_{11} & a_{12} & a_{13} \\\\
a_{21} & a_{22} & a_{23}
\end{array}
\right]
\quad\Rightarrow\quad
\mathbf{A}^\top =
\left[
\begin{array}{cc}
a_{11} & a_{21} \\\\
a_{12} & a_{22} \\\\
a_{13} & a_{23}
\end{array}
\right]
$$


The **identity**  matrix $\mathbf{I}\in\mathbb{R}^{n\times n}$ is defined by a matrix whose diagonal are ones with all other elements being zeros:
$$
\mathbf{I} =
\left[
\begin{array}{cccc}
1 & 0 & 0 & 0 \\\\
0 & 1 & 0 & 0 \\\\
0 & 0 & \ddots & 0 \\\\
0 & 0 & 0 & 1
\end{array}
\right].
$$


The identity matrix represents the linear map that leaves every vector unchanged and acts as the same way that $1$ does in the rational numbers:
$$
\mathbf{I}\mathbf{x} = \mathbf{x}, \qquad
\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}.
$$

!!! note
    Identity matrices appear implicitly in [residual connections](../../notebooks/06_batchnorm_resnet) and linear solvers. Adding $\mathbf{I}$ to a matrix corresponds to biasing a transformation toward preserving information.

The matrix **inversion** provides a formal way to solve linear systems of the form $\mathbf{y} = \mathbf{A}\mathbf{x}$. If the matrix $\mathbf{A}$ is square ($n \times n$) and [invertible](https://en.wikipedia.org/wiki/Invertible_matrix), there exists a matrix $\mathbf{A}^{-1}$ such that
$\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}.$
Multiplying both sides of the equation by $\mathbf{A}^{-1}$ yields $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.$ 

!!! note
    Matrix inversion corresponds to undoing a linear transformation: applying $\mathbf{A}^{-1}$ reverses the effect of $\mathbf{A}$. In practice, however, explicit matrix inversion is rarely used in numerical computation or deep learning. It is primarily a theoretical tool. Solving linear systems is typically done using more stable and efficient methods that avoid forming $\mathbf{A}^{-1}$ directly, especially when matrices are large or ill-conditioned.

## Vector and Matrix Multiplications

Linear algebra uses small set of multiplication rules which make sure that the initial axioms are followed. In deep learning, nearly every forward and backward computation reduces to combinations of the operations described here.

**Dot (inner) product.** For $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$,
$
\mathbf{x}\cdot\mathbf{y}=\sum_{i=1}^n x_i y_i .
$
The result is a scalar. Algebraically, the dot product is [bilinear](https://en.wikipedia.org/wiki/Bilinear_map): linear in each argument when the other is held fixed[^1]. This property is essential for gradient-based optimization. 

!!! note
    The dot product has several complementary interpretations. It measures how strongly $\mathbf{x}$ aligns with weights $\mathbf{y}$ by summing componentwise contributions. Geometrically (when visualization is possible), it measures alignment between vectors: large positive values indicate similar directions, values near zero indicate near-orthogonality, and negative values indicate opposing directions. In practice, it appears as neuron pre-activations, similarity scores, attention mechanisms (queries-keys), and projections.

The dot product between vectors can also be written in matrix form as
$\mathbf{x}\cdot\mathbf{y} = \mathbf{x}^\top \mathbf{y}$, and is commutative: $\mathbf{x}^\top \mathbf{y} = \mathbf{y}^\top \mathbf{x}.$ 

!!! note
    Orientation matters. $\mathbf{x}\mathbf{y}^\top$ and $\mathbf{x}^\top\mathbf{y}$ are different objects with different meanings. Many shape errors in neural network implementations come from ignoring this distinction.

**Hadamard (elementwise) product.** The Hadamard product multiplies vectors componentwise:
$
(\mathbf{x}\odot\mathbf{y})_i=x_i y_i .
$
The result is a vector in $\mathbb{R}^n$. This operation does not mix coordinates and is not a linear map. Despite this, it is occasionally used in deep learning, for example, when applying attention masks or feature-wise scaling.

!!! note
    The **cross product** known from physics curriculum is defined only in $\mathbb{R}^3$ (and, with [special constructions](https://en.wikipedia.org/wiki/Seven-dimensional_cross_product), $\mathbb{R}^7$). It produces a vector orthogonal to its inputs and relies on three-dimensional geometry. Since deep learning representations typically live in high-dimensional spaces with no notion of "orthogonal direction in space", the cross product has no general role in deep learning and is not used in standard models.

**Matrix-vector multiplication.** Let $\mathbf{A}\in\mathbb{R}^{m\times n}$ and $\mathbf{x}\in\mathbb{R}^n$. Then
$$
\mathbf{y}=\mathbf{A}\mathbf{x}\in\mathbb{R}^m,\qquad
y_i=\sum_{j=1}^n a_{ij}x_j .
$$
Each output component is a _dot product_ between one row of $\mathbf{A}$ and $\mathbf{x}$. This is the fundamental linear operation in deep learning: rows act as learned feature detectors and dimensionality may change ($n\to m$). 

!!! note
    A [fully connected  layer](../../notebooks/02_neural_network) has the form $\mathbf{y}=\mathbf{W}\mathbf{x}+\mathbf{b}$. The nonlinearity that follows does not alter the linearity of this step.

**Matrix-matrix multiplication.** For $\mathbf{A}\in\mathbb{R}^{m\times n}$ and $\mathbf{B}\in\mathbb{R}^{n\times p}$,
$$
\mathbf{C}=\mathbf{A}\mathbf{B}\in\mathbb{R}^{m\times p},\qquad
c_{ij}=\sum_{k=1}^n a_{ik}b_{kj}.
$$
This represents composition of linear maps: applying $\mathbf{B}$ then $\mathbf{A}$ equals applying $\mathbf{A}\mathbf{B}$. Stacked linear layers, gradient propagation via transposes, and backpropagation all rely on this structure.

!!! warning "Important"
    Matrix multiplication satisfies _distributivity_ and _associativity_, but it is **not** _commutative_ $\mathbf{A}\mathbf{B}\neq\mathbf{B}\mathbf{A}$. This reflects the fact that matrix multiplication represents the composition of linear transformations. Changing the order changes which transformation is applied first, and therefore changes the result. Only in special cases—when two transformations are compatible in a specific way—does commutativity hold.

Note that, for any compatible matrices:
$(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top .$
The order reverses because transposition swaps rows and columns, effectively reversing the sequence of linear transformations. This property is used constantly in backpropagation, where gradients are propagated through layers via transposed weight matrices.

!!! note
    Since a scalar is equal to its own transpose, this identity also explains why the dot product is commutative. Written in matrix form:
    $\mathbf{x}^\top \mathbf{y} = (\mathbf{x}^\top \mathbf{y})^\top = \mathbf{y}^\top \mathbf{x}.$ What appears as a symmetry of vectors is therefore a direct consequence of more general properties of matrix transpose.

## Linear Dependence and Span

A collection of vectors is **linearly dependent** if at least one vector in the set can be written as a linear combination of the others. Formally, vectors $\mathbf{v}_1,\dots,\mathbf{v}_k$ are linearly dependent if there exist scalars $\alpha_1,\dots,\alpha_k$, not all zero, such that
$$
\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0}.
$$
Linear dependence means redundancy: some vectors do not add new directions or information. If no such non-trivial combination exists, the vectors are **linearly independent**.

!!! note
    In deep learning and applied linear algebra, linear dependence indicates unnecessary or duplicated features. Independent vectors represent genuinely distinct directions in a space.

The **span** of a set of vectors is the collection of all vectors that can be formed by taking linear combinations of them. Given vectors $\mathbf{v}_1,\dots,\mathbf{v}_k$, their span consists of all vectors that can be written as
$$
\mathbf{s} = [\,\mathbf{v}_1\ \mathbf{v}_2\ \cdots\ \mathbf{v}_k\,]\boldsymbol{\alpha},
\qquad \boldsymbol{\alpha}\in\mathbb{R}^k.
$$

The span describes all vectors that are reachable using those directions. If the vectors are linearly dependent, their span does not grow when all vectors are included. Dependent vectors do not expand the space.

!!! note
    In practice, the span corresponds to the set of outputs a linear layer can produce. Linear dependence among columns of a weight matrix limits expressiveness, while linear independence maximizes the range of representable transformations.





[^1]: A function $f(\mathbf{x},\mathbf{y})$ is called **bilinear** if it is linear in each argument separately when the other argument is held fixed. For the dot product $f(\mathbf{x},\mathbf{y})=\mathbf{x}\cdot\mathbf{y}$, this means:
Holding $\mathbf{y}$ fixed, the map $\mathbf{x}\mapsto \mathbf{x}\cdot\mathbf{y}$ is linear:
$$
(\alpha \mathbf{x}_1+\beta \mathbf{x}_2)\cdot\mathbf{y}
= \alpha(\mathbf{x}_1\cdot\mathbf{y})+\beta(\mathbf{x}_2\cdot\mathbf{y}).
$$
Holding $\mathbf{x}$ fixed, the map $\mathbf{y}\mapsto \mathbf{x}\cdot\mathbf{y}$ is also linear:
$$
\mathbf{x}\cdot(\alpha \mathbf{y}_1+\beta \mathbf{y}_2)
= \alpha(\mathbf{x}\cdot\mathbf{y}_1)+\beta(\mathbf{x}\cdot\mathbf{y}_2).
$$
Bilinearity does not mean the function is linear in both arguments at once. It means that if one vector is treated as constant, the dot product behaves exactly like a linear function of the other. This property is what allows dot products to distribute over sums and pull out scalar factors, and it is why gradients propagate cleanly through linear layers in deep learning.