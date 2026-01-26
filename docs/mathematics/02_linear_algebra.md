# Linear Algebra

<div style="margin:.3rem 0 1rem;font-size:.9em;color:#555;display:flex;align-items:center;gap:.35rem;font-family:monospace">
  <time datetime="2026-01-26">26 Jan 2026</time>
</div>

Linear algebra is the branch of mathematics that studies vector spaces and the linear mappings between them. In deep learning, almost all computation is formulated in the language of linear algebra: data, model parameters, activations, gradients are represented as vectors or matrices. A clear understanding of what these objects represent — and how they behave under linear operations — is necessary not only for correct implementation, but for reasoning about model structure, learning dynamics, and numerical behavior.

!!! info
    The following sources were consulted in preparing this material:

    - Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Chapter 2: Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html).
    - Sanderson, G. *Essence of Linear Algebra*. 3Blue1Brown. <https://www.3blue1brown.com/topics/linear-algebra>

!!! warning "Important"
    Please note that some concepts in this material are simplified for pedagogical purposes. These simplifications slightly reduce precision but preserve the core ideas relevant to deep learning.

## Scalars and Vectors

A [scalar](https://en.wikipedia.org/wiki/Scalar_(mathematics)) is a single number (often real-valued). It is more challenging to define a **vector**. From a mathematician's point of view, a vector is an element of a [vector space](https://en.wikipedia.org/wiki/Vector_space): something you can **add and scale while satisfying certain axioms** (closure, associativity, distributivity, etc.). The axioms exist to guarantee that linear combinations behave predictably. From these requirements, any [linear map](https://en.wikipedia.org/wiki/Linear_map) $f$ between vector spaces satisfies the following combined property of additivity and homogeneity (scaling):
$$
f(\alpha x + \beta y) = \alpha f(x) + \beta f(y).
$$
This equation does not define vectors themselves, but rather characterizes [linear](https://en.wikipedia.org/wiki/Linearity) transformations acting on vectors. Vectors are defined by the operations of addition and scalar multiplication and linear maps are functions that preserve this structure.

!!! note
    Raw audio signals satisfy the linearity properties to a good approximation. If two sounds are played at the same time, the resulting waveform is (approximately) the sum of the individual waveforms. If the volume of a sound is increased or decreased, its waveform is scaled by a constant factor. Because audio combines by superposition and scales linearly with amplitude, it can be naturally represented as a vector and manipulated using linear algebra.

From a physicist's point of view, a vector represents a **quantity with direction and magnitude** (e.g. velocity, force). You add forces, scale forces, decompose into components. The vector predicts physical behavior. Lastly, from a computer scientist's point of view, a vector is simply an **array of numbers**. It can represent pixel values of an image, coordinates of a point, words in a document, etc.

!!! warning "Important"
     In many ways, machine/deep learning borrows terminology from mathematics and uses it rather freely. Terms like **vector**, **dimension**, **space**, **metric**, **manifold**, and even **linear** are frequently misused. For example, by "dimension" one could assume "vector size". This is convenient shorthand, but it can break intuition if you don't keep in mind the underlying differences between deep learning and mathematics which can use the same tools for different purposes.

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

In deep learning, such a vector is typically understood operationally: it is stored as a contiguous array of \(n\) real numbers and is mathematically an element of \(\mathbb{R}^n\), the [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) of \(\mathbb{R}\) with itself \(n\) times. In this context, its "dimension" refers simply to its length \(n\). When \(n = 2\) or \(n = 3\), the vector can be visualized geometrically as a point or an arrow. When $n$ is large, direct visualization is no longer possible, but the same algebraic operations — addition, scalar multiplication, dot products, and linear transformations — still apply. 

!!! note
    Depending on context, vectors can be visualized in different ways. In geometry and physics, they are often drawn as arrows representing magnitude and direction. In other settings, a vector can be viewed as a function that assigns a value to each index or coordinate. These visualizations are useful for building intuition, especially in low dimensions, but they do not alter the underlying algebraic definition of a vector. Linear algebra itself does not rely on geometric interpretation. It is fundamentally an algebraic theory of vector spaces and linear maps, and all definitions and results are independent of visualization. Geometry serves only as an intuitive aid not as a prerequisite. Beyond three dimensions, geometry in the visual sense becomes unusable. Since most representations in deep learning live in very high-dimensional spaces, geometric visualization is generally not available and plays no direct role in practice. What remains meaningful are algebraic and analytical notions — such as inner products, norms, projections, and linear maps — rather than pictures or spatial intuition.

## Matrices and Tensors

A [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)) is a rectangular array of numbers arranged in rows and columns. Formally, a real-valued matrix with \(m\) rows and \(n\) columns is an element of \(\mathbb{R}^{m \times n}\):
$$
\mathbf{A} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\\\
a_{21} & a_{22} & \cdots & a_{2n} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

From a mathematical point of view, a matrix represents a linear map between vector spaces. Given a vector \(\mathbf{x} \in \mathbb{R}^n\), multiplication by a matrix \(\mathbf{A} \in \mathbb{R}^{m \times n}\) produces a new vector \(\mathbf{y} \in \mathbb{R}^m\): $$
\mathbf{A}\mathbf{x} = \mathbf{y}.$$ 

This operation encodes all **linear transformations**: rotations, scalings, projections, and combinations of these. The key idea is that matrices do not just store numbers; they describe how vectors are transformed. In deep learning, matrices appear everywhere. For example, in

- Model parameters (weights of fully connected layers)
- Batches of input data
- Linear layers of the form \( \mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b} \)
- [Jacobians and Hessians](../01_calculus) (implicitly, through [automatic differentiation](../../notebooks/01_backprop))

!!! warning "Important"
    In deep learning, layers of the form $\mathbf{W}\mathbf{x}+\mathbf{b}$ are often informally called "linear". This shorthand is convenient in practice, but it is useful to remember that the bias term shifts the output and allows models to represent functions that a purely linear map could not. Strictly speaking, the map $\mathbf{x}\mapsto \mathbf{W}\mathbf{x}$ is linear: it preserves addition, scaling, and maps the zero vector to the zero vector. Adding a bias term $\mathbf{b}$ produces an [affine map](https://en.wikipedia.org/wiki/Affine_transformation), which is a linear transformation followed by a translation. Because of this translation, affine maps do not preserve the origin.

A matrix is stored as a 2D array in memory, but it should be understood as a single object representing a linear operation. Confusing these two viewpoints — matrix as data vs. matrix as transformation — is a common source of misunderstanding.

!!! note 
    When training [neural networks](../../notebooks/02_neural_network), we rarely reason about individual entries of a matrix. Instead, we reason about the effect of the matrix as a whole: how it mixes input features, how it changes dimensionality, and how it interacts with nonlinearities. Frameworks exploit this by implementing matrix multiplication using highly optimized numerical kernels.

A [tensor](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) is a generalization of scalars, vectors, and matrices to higher dimensions. Informally: a scalar is a 0-order tensor, a vector is a 1-order tensor, a matrix is a 2-order tensor, etc. In deep learning practice, a tensor is best understood as a multidimensional array of numbers with a fixed shape.

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
Multiplying both sides of the equation by $\mathbf{A}^{-1}$ yields
$\mathbf{x}=\mathbf{A}^{-1}\mathbf{y}.$

!!! note
    Matrix inversion corresponds to undoing a linear transformation: applying $\mathbf{A}^{-1}$ reverses the effect of $\mathbf{A}$. In practice, however, explicit matrix inversion is rarely used in numerical computation or deep learning. It is primarily a theoretical tool. Solving linear systems is typically done using more stable and efficient methods that avoid forming $\mathbf{A}^{-1}$ directly, especially when matrices are large or ill-conditioned.

## Vector and Matrix Multiplications

Linear algebra uses small set of multiplication rules which make sure that the initial axioms are followed. In deep learning, nearly every forward and backward computation reduces to combinations of the operations described here.

**Dot (inner) product.** For $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$,
$
\mathbf{x}\cdot\mathbf{y}=\sum_{i=1}^n x_i y_i .
$
The result is a scalar. Algebraically, the dot product is [bilinear](https://en.wikipedia.org/wiki/Bilinear_map): linear in each argument when the other is held fixed[^bilinear]. This property is essential for gradient-based optimization. 

!!! note
    The dot product has several complementary interpretations. It measures how strongly $\mathbf{x}$ aligns with weights $\mathbf{y}$ by summing componentwise contributions. Geometrically (when visualization is possible), it measures alignment between vectors: large positive values indicate similar directions, values near zero indicate near-orthogonality, and negative values indicate opposing directions. In practice, it appears as neuron pre-activations, similarity scores, attention mechanisms (queries-keys), and projections.

The dot product between vectors can also be written in matrix form as
$\mathbf{x}\cdot\mathbf{y} = \mathbf{x}^\top \mathbf{y}$, and is commutative: $\mathbf{x}^\top \mathbf{y} = \mathbf{y}^\top \mathbf{x}.$ 

!!! note
    Orientation matters. $\mathbf{x}\mathbf{y}^\top$ and $\mathbf{x}^\top\mathbf{y}$ are different objects with different meanings. Many shape errors in neural network implementations come from ignoring this distinction.

**Hadamard (elementwise) product.** The Hadamard product multiplies vectors componentwise:
$
(\mathbf{x}\odot\mathbf{y})_i = x_i y_i .
$
The result is a vector in $\mathbb{R}^n$. This operation does not mix coordinates: each output component depends only on the corresponding input components. For a fixed vector, it acts as a simple coordinate-wise scaling. In deep learning, the Hadamard product is used when features are masked, gated, or rescaled individually, such as in attention masks.

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

Consider the linear system $\mathbf{A}\mathbf{x}=\mathbf{b}$
with $\mathbf{A}\in\mathbb{R}^{m\times n}$ and $\mathbf{b}\in\mathbb{R}^m$. The system has a solution iff $\mathbf{b}$ lies in the span of the columns of $\mathbf{A}$. This is called the **column space** (or range) of $\mathbf{A}$. For the system to have a solution for all $\mathbf{b}\in\mathbb{R}^m$, the column space of $\mathbf{A}$ must be all of $\mathbb{R}^m$. This immediately requires $n\ge m$. Otherwise, the column space has dimension at most $n<m$ and cannot fill $\mathbb{R}^m$. 

!!! note
    For example, a $3\times2$ matrix can only produce a 2-dimensional plane inside $\mathbb{R}^3$. The equation has a solution only when $\mathbf{b}$ lies on that plane.

The condition $n\ge m$ is necessary but not sufficient. Columns may be redundant. A matrix whose columns are linearly dependent does not expand its column space. Therefore, for the column space to equal $\mathbb{R}^m$, the matrix must have rank $m$, meaning that it contains $m$ linearly independent columns. This condition is necessary and sufficient for $\mathbf{A}\mathbf{x}=\mathbf{b}$ to have a solution for every $\mathbf{b}\in\mathbb{R}^m$.

!!! success "Exercise"
    Which properties must a matrix have in order to ensure **uniqueness** of the solution?

A square matrix with linearly dependent columns is called a **singular** matrix. If $\mathbf{A}$ is invertible, the unique solution is
$\mathbf{x}=\mathbf{A}^{-1}\mathbf{b}.$[^inverse] If $\mathbf{A}$ is not square or is singular, solutions may still exist, but matrix inversion cannot be used.

Closely related to the column space is the **null space** of a matrix. The null space of $\mathbf{A}$ is the set of all vectors $\mathbf{x}$ such that
$
\mathbf{A}\mathbf{x} = \mathbf{0}.
$
Vectors in the null space are mapped to zero and therefore cannot be recovered from the output. A matrix has linearly dependent columns iff its null space contains nonzero vectors.

## Linear Systems

Consider the [linear system](https://en.wikipedia.org/wiki/Linear_system)
$\mathbf{A}\mathbf{x}=\mathbf{b},$ where $\mathbf{A}\in\mathbb{R}^{m\times n}$. A solution _exists_ iff the vector $\mathbf{b}$ lies in the column space of $\mathbf{A}$. If $\mathbf{b}$ cannot be expressed as a linear combination of the columns of $\mathbf{A}$, then no vector $\mathbf{x}$ can satisfy the equation.

If a solution exists, its _uniqueness_ depends on the null space of $\mathbf{A}$. If the null space contains only the [zero vector](https://en.wikipedia.org/wiki/Null_vector), then the solution is unique. In this case, no nonzero direction can be added to a solution without changing the output.

If the null space contains nonzero vectors, then _infinitely many solutions_ exist. Any solution can be modified by adding a null-space vector, producing a different input that yields the same output. In this situation, the linear map collapses information: different inputs are indistinguishable after applying $\mathbf{A}$.

!!! note
    In elementary linear algebra, these cases are typically analyzed using [Gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) and row-echelon form. While these procedures are essential for conceptual understanding and small problems, they are not used directly in deep learning or large-scale numerical computation. Modern frameworks instead rely on matrix factorizations and optimized solvers that achieve the same goals more efficiently and with better numerical stability.


## Basis and Rank

A **basis** of a vector space is a set of vectors that is both linearly independent and spanning the space. Every vector in the space can be written _uniquely_ as a linear combination of the basis vectors. Consider the standard basis of $\mathbb{R}^2$, given by
$\mathbf{e}_1=[\,1\;\;0\,]^\top$ and $\mathbf{e}_2=[\,0\;\;1\,]^\top$.
Any vector $\mathbf{x}\in\mathbb{R}^2$ can be written uniquely as
$$
\mathbf{x}=x_1\mathbf{e}_1+x_2\mathbf{e}_2.
$$
The pair $(x_1,x_2)$ are the coordinates of $\mathbf{x}$ in the standard basis. Now consider a different basis,
$\mathbf{v}_1=[\,1\;\;1\,]^\top$ and $\mathbf{v}_2=[\,1\;\;-1\,]^\top$.
This set is linearly independent and spans $\mathbb{R}^2$.
The same vector $\mathbf{x}$ can be written as
$$
\mathbf{x}=c_1\mathbf{v}_1+c_2\mathbf{v}_2,
$$
but the coefficients $(c_1,c_2)$ are different. The vector itself has not changed, only its coordinates have.

!!! note
    Let $\mathbf{V}=[\,\mathbf{v}_1\ \mathbf{v}_2\,]$.
    Then $\mathbf{x}=\mathbf{V}\mathbf{c}$ and $\mathbf{c}=\mathbf{V}^{-1}\mathbf{x}$.
    Changing basis corresponds to switching between coordinate systems using $\mathbf{V}$ and its inverse.

If a space has a basis consisting of $k$ vectors, we say the space has **dimension $k$**. In $\mathbb{R}^n$, any basis contains exactly $n$ vectors. For matrices, the analogous notion to dimension is **rank**. The rank of a matrix is the dimension of the space spanned by its columns (equivalently, its rows). It measures how many linearly independent directions the matrix preserves. If a matrix has rank $r$, then its columns form a basis for an $r$-dimensional subspace. 

Consider the matrix
$\mathbf{A}=
\begin{bmatrix}
1 & 1 \\
2 & 2
\end{bmatrix}$.
The second row is a multiple of the first, so $\mathbf{A}$ has rank $1$.

Now take two different vectors,
$\mathbf{x}_1 = [\,1\;\;0\,]^\top$
and
$\mathbf{x}_2 = [\,0\;\;1\,]^\top$.
They are clearly distinct. Multiplying by $\mathbf{A}$ gives
$\mathbf{A}\mathbf{x}_1 = [\,1\;\;2\,]^\top$
and
$\mathbf{A}\mathbf{x}_2 = [\,1\;\;2\,]^\top$. Although $\mathbf{x}_1 \neq \mathbf{x}_2$, they are mapped to the same output. The matrix cannot distinguish between directions that differ only within its null space. Information is collapsed because the transformation preserves only one independent direction.


!!! note
    In deep learning, rank determines whether a linear layer preserves information or collapses it into a lower-dimensional representation. 

## Norms

Sometimes we need to measure the size of a vector. In machine learning, this is usually done using a **norm**. Formally, the $L_p$ norm of a vector $\mathbf{x}\in\mathbb{R}^n$ is defined as
$$
\|\mathbf{x}\|_p=\left(\sum_i |x_i|^p\right)^{1/p},
\qquad p\ge 1.
$$

Intuitively, a norm measures the distance from the origin to the point $\mathbf{x}$. More precisely, a norm is any function $f$ satisfying:

- $f(\mathbf{x})=0 \Rightarrow \mathbf{x}=\mathbf{0}$
- $f(\mathbf{x}+\mathbf{y})\le f(\mathbf{x})+f(\mathbf{y})$ (triangle inequality)
- $f(\alpha\mathbf{x})=|\alpha|f(\mathbf{x})$ for all $\alpha\in\mathbb{R}$

The $L_2$ norm ($p=2$), called the [Euclidean norm](https://en.wikipedia.org/wiki/Euclidean_distance), is used so frequently that it is often written simply as $\|\mathbf{x}\|$. Its square can be written compactly as
$
\|\mathbf{x}\|_2^2=\mathbf{x}^\top\mathbf{x}.
$

!!! note
    The **squared** $L_2$ norm is often preferred in optimization because it is smooth and has simple derivatives. 

However, the squared $L_2$ norm grows slowly near zero, which makes it less suitable when distinguishing exact zeros from small nonzero values matters. In such cases, the $L_1$ norm, also known as the [Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry), is commonly used:
$
\|\mathbf{x}\|_1=\sum_i |x_i|
$. Here, each component contributes linearly, so moving an element away from zero by $\varepsilon$ increases the norm by exactly $\varepsilon$. This property makes the $L_1$ norm useful for encouraging sparsity.

!!! note
    Sometimes one wants to count the number of nonzero entries in a vector. This quantity is often (incorrectly) called the $L_0$ norm. It is not a true norm, because it is invariant under scaling (doesn't meet the third property described previously). In practice, the $L_1$ norm is often used as a continuous [surrogate](https://en.wikipedia.org/wiki/Surrogate_model) for this count.

Another common norm is the $L_\infty$ norm ([maximum norm](https://en.wikipedia.org/wiki/Uniform_norm))[^maxnorm], defined as
$
\|\mathbf{x}\|_\infty=\max_i |x_i|.
$
It measures the magnitude of the largest component of the vector.

Finally, the dot product of two vectors can be expressed in terms of norms:
$
\mathbf{x}^\top\mathbf{y}=\|\mathbf{x}\|_2\,\|\mathbf{y}\|_2\cos\theta,
$
where $\theta$ is the angle between $\mathbf{x}$ and $\mathbf{y}$. This relation explains why the dot product measures both magnitude and alignment, and why normalized dot products are often used as similarity measures in machine/deep learning.

!!! note
    Norms can also be defined for matrices. In deep learning, the most common choice is the [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm), which is directly analogous to the $L_2$ norm of a vector.

## Diagonal, Symmetric, Orthogonal Matrices

A **diagonal matrix** consists of zeros everywhere except possibly on the main diagonal. Formally, a matrix $\mathbf{D}$ is diagonal if $D_{ij}=0$ for all $i\neq j$. The identity matrix is a special case of a diagonal matrix with all diagonal entries equal to 1. We write $\mathrm{diag}(\mathbf{v})$ to denote a square diagonal matrix whose diagonal entries are given by the vector $\mathbf{v}$.

Diagonal matrices are important because multiplication by them is computationally efficient. The product $\mathrm{diag}(\mathbf{v})\mathbf{x}$ simply scales each component of $\mathbf{x}$ by the corresponding entry of $\mathbf{v}$: 
$$
\mathrm{diag}(\mathbf{v})\mathbf{x} = \mathbf{v}\odot\mathbf{x}.
$$

Inversion is also efficient. A square diagonal matrix is invertible iff all diagonal entries are nonzero, in which case
$$
\mathrm{diag}(\mathbf{v})^{-1} = \mathrm{diag}\bigl([1/v_1,\dots,1/v_n]^\top\bigr).
$$

!!! note
    Many algorithms can be simplified and accelerated by restricting certain matrices to be diagonal.

A **symmetric matrix** is a matrix equal to its transpose:
$
\mathbf{A}=\mathbf{A}^\top.
$
Symmetric matrices often arise when entries depend on pairs of elements in an order-independent way, such as distance matrices where $A_{ij}=A_{ji}$. A symmetric matrix $\mathbf{A}$ is called _positive semidefinite_ if
$\mathbf{x}^\top\mathbf{A}\mathbf{x}\ge 0$ for all $\mathbf{x}$, and _positive definite_ if the inequality is strict for all $\mathbf{x}\neq 0$.

!!! note
    These notions describe how a matrix assigns a scalar value to every direction via the quadratic form $\mathbf{x}^\top\mathbf{A}\mathbf{x}$. Matrices of the form $\mathbf{A}^\top\mathbf{A}$ and [covariance](../03_probability) matrices are always positive semidefinite. When a matrix is positive definite, the expression $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ behaves like a squared norm.

Two vectors $\mathbf{x}$ and $\mathbf{y}$ are orthogonal if $\mathbf{x}^\top\mathbf{y}=0$. In $\mathbb{R}^n$, at most $n$ nonzero vectors can be mutually orthogonal. If the vectors are both orthogonal and have unit norm[^unitvector], they are called **orthonormal**. An **orthogonal matrix** is a square matrix whose rows and columns are orthonormal:
$$
\mathbf{A}^\top\mathbf{A}=\mathbf{A}\mathbf{A}^\top=\mathbf{I}.
$$
This implies
$
\mathbf{A}^{-1}=\mathbf{A}^\top,
$
so orthogonal matrices are especially convenient computationally. 

!!! note
    Pay attention that "orthogonal" here means "orthonormal"; there is no standard term for matrices whose rows or columns are orthogonal but not normalized.


## Eigenvalues and Eigenvectors

Let $\mathbf{A}\in\mathbb{R}^{n\times n}$. A nonzero vector $\mathbf{v}$ is called an **eigenvector** of $\mathbf{A}$ if there exists a scalar $\lambda$ such that
$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$
The scalar $\lambda$ is the corresponding **eigenvalue**. 

Illustrated below is the impact of the transformation matrix $\mathbf{A}=\begin{bmatrix}2&1\\1&2\end{bmatrix}$ on different vectors.

<figure>
  <img src="../../assets/images/linear_algebra/eigenvectors.gif" alt="Eigenvectors and eigenvlues" style="max-width: 100%; height: auto;">
  <figcaption style="margin-top: 0.5em; font-size: 0.9em; opacity: 0.85;">
    The transformation matrix preserves the directions of the magenta vectors parallel to $\mathbf{v}_{\lambda=1}=[\,1\;\;-1\,]^\top$ and the blue vectors parallel to $\mathbf{v}_{\lambda=3}=[\,1\;\;1\,]^\top$. Red vectors are not parallel to either eigenvector, so their directions change under the transformation. Magenta vectors keep the same length (eigenvalue $1$), while blue vectors become three times longer (eigenvalue $3$). By <a href="//commons.wikimedia.org/wiki/User:LucasVB" title="User:LucasVB">Lucas Vieira</a> - <span class="int-own-work" lang="en">Own work</span>, Public Domain, <a href="https://commons.wikimedia.org/w/index.php?curid=19449791">Link</a>
  </figcaption>
</figure>

Eigenvectors identify directions that are preserved by the linear transformation $\mathbf{A}$. Applying $\mathbf{A}$ to an eigenvector does not change its direction; it only scales it by the factor $\lambda$. If $|\lambda|>1$, vectors in that direction are stretched, if $|\lambda|<1$, they are compressed, if $\lambda=0$, the direction is collapsed.

## Determinant

The [determinant](https://en.wikipedia.org/wiki/Determinant) of a square matrix $\mathbf{A}$, denoted $\det(\mathbf{A})$, is a scalar that summarizes how the linear transformation defined by $\mathbf{A}$ scales space. The determinant is equal to the product of the eigenvalues of $\mathbf{A}$:
$$
\det(\mathbf{A})=\prod_{i=1}^n \lambda_i.
$$
This interpretation becomes especially clear when $\mathbf{A}$ is diagonal or diagonalizable. For a diagonal matrix
$
\boldsymbol{\Lambda}=\mathrm{diag}(\lambda_1,\dots,\lambda_n),
$
the determinant is simply
$$
\det(\boldsymbol{\Lambda})=\lambda_1\lambda_2\cdots\lambda_n.
$$

Each diagonal entry scales space along one coordinate direction, and the determinant multiplies these scalings together. The following transformation scales area by a factor of 6.
$$
\boldsymbol{\Lambda}
=
\begin{bmatrix}
2 & 0 \\\\
0 & 3
\end{bmatrix}
\;\Rightarrow\;
\det(\boldsymbol{\Lambda}) = 2\cdot 3 = 6.
$$


!!! success "Exercise"
    Let $\mathbf{A}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ be the eigendecomposition of a real symmetric matrix, where $\mathbf{Q}$ is orthogonal and $\boldsymbol{\Lambda}=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$. Show that
    $$
    \det(\mathbf{A})=\det(\boldsymbol{\Lambda})=\prod_i \lambda_i,
    $$
    and explain why the orthogonality of $\mathbf{Q}$ implies that the change of basis does not affect volume.


Geometrically, the absolute value $|\det(\mathbf{A})|$ measures how much the transformation expands or contracts volume. If $|\det(\mathbf{A})|>1$, volume is expanded; if $|\det(\mathbf{A})|<1$, volume is contracted. If $\det(\mathbf{A})=1$, volume is preserved. If $\det(\mathbf{A})=0$, space is collapsed along at least one direction, causing the transformation to lose volume entirely. In this case, the matrix is **singular** and **not invertible**.

!!! note
    The determinant can also be defined directly in terms of matrix entries, without reference to eigenvalues. For example, for a $2\times2$ matrix,
    $$
    \mathbf{A}
    =
    \begin{bmatrix}
    a & b \\\\
    c & d
    \end{bmatrix},
    \qquad
    \det(\mathbf{A}) = ad - bc.
    $$
    For larger matrices, the determinant is defined recursively via [cofactor expansion](https://en.wikipedia.org/wiki/Laplace_expansion) or computed using row operations. While these formulas are often used for computation, the eigenvalue interpretation provides the clearest conceptual understanding of what the determinant represents for deep learning.


## Eigendecomposition

If a matrix $\mathbf{A}$ has a full set of linearly independent eigenvectors, these can be arranged as columns of a matrix $\mathbf{V}$, with the corresponding eigenvalues placed on the diagonal of a matrix $\boldsymbol{\Lambda}$. Such a factorization is called the [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) and the matrix can then be written as
$$
\mathbf{A}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}.
$$

Eigendecomposition represents a linear transformation as:

1. a change of basis into the eigenvector basis ($\mathbf{V}^{-1}$),
2. independent scalings along each eigenvector direction ($\boldsymbol{\Lambda}$),
3. a change back to the original basis ($\mathbf{V}$).

Eigendecomposition is also the cleanest example of a general computational principle: decompose a matrix into parts with simple structure. Suppose $\mathbf{A}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$. Then:
$$
\begin{aligned}
\mathbf{A}^2
&= (\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1})
   (\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}) \\\\
&= \mathbf{V}\boldsymbol{\Lambda}
   (\mathbf{V}^{-1}\mathbf{V})
   \boldsymbol{\Lambda}\mathbf{V}^{-1} \\\\
&= \mathbf{V}\boldsymbol{\Lambda}^2\mathbf{V}^{-1}.
\end{aligned}
$$

The middle factors $\mathbf{V}^{-1}\mathbf{V}$ cancel to the identity. Repeating this argument gives
$$
\mathbf{A}^k=\mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1}.
$$

Thus, repeated matrix multiplication reduces to raising the diagonal entries $\lambda_i$ to the power $k$:
$$
\boldsymbol{\Lambda}^k
=\mathrm{diag}(\lambda_1,\dots,\lambda_n)^k
=\mathrm{diag}(\lambda_1^k,\dots,\lambda_n^k).
$$
Decomposition replaces repeated dense matrix multiplication with exponentiating scalars $\lambda_i$. The expensive part—the interaction between coordinates—disappears: in the eigenvector basis, each component is scaled independently by $\lambda_i^k$.

!!! note
    Diagonal matrices are cheap to multiply, invert, and exponentiate, which is why eigendecomposition is so effective.

Not every matrix admits eigendecomposition with real eigenvalues and eigenvectors. However, every _real symmetric matrix_ does:
$$
\mathbf{A}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top,
$$
where $\mathbf{Q}$ is orthogonal and $\boldsymbol{\Lambda}$ is real and diagonal. In this case, the inverse is especially simple: $\mathbf{Q}^{-1}=\mathbf{Q}^\top$.

The eigendecomposition of a real symmetric matrix is not necessarily unique. When eigenvalues are repeated, any orthonormal basis of the corresponding eigenspace yields a valid decomposition. By convention, eigenvalues are usually ordered from largest to smallest.

Eigenvalues immediately reveal important properties. A matrix is **singular** iff at least one eigenvalue is zero. For symmetric matrices, eigenvalues also characterize quadratic forms:
$$
\mathbf{x}^\top\mathbf{A}\mathbf{x}, \qquad \|\mathbf{x}\|_2=1.
$$
The maximum and minimum values are the largest and smallest eigenvalues, attained at the corresponding eigenvectors. 

## Singular Value Decomposition

Eigendecomposition is defined only for square matrices. For a general matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$, the analogous tool is the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition):
$$
\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top,
$$
where $\mathbf{U}\in\mathbb{R}^{m\times m}$ and $\mathbf{V}\in\mathbb{R}^{n\times n}$ are orthogonal matrices, and $\boldsymbol{\Sigma}\in\mathbb{R}^{m\times n}$ is diagonal (not necessarily square). The diagonal entries of $\boldsymbol{\Sigma}$ are called the **singular values** of $\mathbf{A}$. The columns of $\mathbf{U}$ are the **left singular vectors**, and the columns of $\mathbf{V}$ are the **right singular vectors**.

<figure>
  <img src="../../assets/images/linear_algebra/svd.svg" alt="Singular Value Decomposition (SVD)" style="max-width: 100%; height: auto;">
  <figcaption style="margin-top: 0.5em; font-size: 0.9em; opacity: 0.85;">
    Illustration of the singular value decomposition $U\Sigma V^{*}$ of a real $2 \times 2$ matrix $M$. Top: The action of $M$, indicated by its effect on the unit disc $D$ and the two canonical unit vectors $e_1$ and $e_2$. Left: The action of $V^{*}$, a rotation, on $D$, $e_1$, and $e_2$. Bottom: The action of $\Sigma$, a scaling by the singular values $\sigma_1$ horizontally and $\sigma_2$ vertically. Right: The action of $U$, another rotation. By <a href="//commons.wikimedia.org/wiki/User:Georg-Johann" title="User:Georg-Johann">Georg-Johann</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/3.0" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=11342212">Link</a>
  </figcaption>
</figure>

Singular value decomposition is closely related to eigendecomposition. The left singular vectors of $\mathbf{A}$ are the eigenvectors of $\mathbf{A}\mathbf{A}^\top$, and the right singular vectors are the eigenvectors of $\mathbf{A}^\top\mathbf{A}$. The nonzero singular values are the square roots of the nonzero eigenvalues of $\mathbf{A}^\top\mathbf{A}$ (and equivalently of $\mathbf{A}\mathbf{A}^\top$).

!!! tip
    For a more detailed explanation on how decompositions emerge, see the supplementary material on the [singular value decomposition](../../supplementary/svd). 

One important use of singular value decomposition is that it provides a principled way to extend matrix inversion to non-square or singular matrices via the pseudoinverse.

## Moore–Penrose Pseudoinverse

Matrix inversion is not defined for non-square matrices. Suppose we want a matrix $\mathbf{B}$ that acts like a left-inverse so that we can solve
$\mathbf{A}\mathbf{x}=\mathbf{y}$
by writing $\mathbf{x}=\mathbf{B}\mathbf{y}$. Depending on the shape of $\mathbf{A}$, an exact solution may not exist (tall matrices) or may not be unique (wide matrices).

The [Moore–Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) $\mathbf{A}^+$ provides a standard choice. It can be defined as
$$
\mathbf{A}^+ = \lim_{\alpha\to 0}(\mathbf{A}^\top\mathbf{A}+\alpha\mathbf{I})^{-1}\mathbf{A}^\top,
$$
but in practice it is computed using the SVD. If $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, then
$$
\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top,
$$
where $\boldsymbol{\Sigma}^+$ is obtained by taking the reciprocal of each nonzero diagonal entry of $\boldsymbol{\Sigma}$ and then transposing the result.

If $\mathbf{A}$ has more columns than rows ($n>m$), the system may have infinitely many solutions. The pseudoinverse returns the solution with minimal Euclidean norm $\|\mathbf{x}\|_2$ among all solutions. If $\mathbf{A}$ has more rows than columns ($m>n$), the system may have no exact solution. In that case, $\mathbf{x}=\mathbf{A}^+\mathbf{y}$ minimizes the least-squares error $\|\mathbf{A}\mathbf{x}-\mathbf{y}\|_2$. 

!!! tip
    To get a nice and clear geometric intuition on the matter, see the chapter on the **method of least squares** from David C. Lay's _[Linear Algebra and its Applications](https://www.google.com/books/edition/_/bM6gBwAAQBAJ?hl=en&newbks=1)_.  
 
## Trace Operator

The **trace** of a square matrix is the sum of its diagonal entries:
$\mathrm{Tr}(\mathbf{A}) = \sum_i A_{ii}.$
The trace is useful because it allows scalar quantities involving matrices to be written compactly using matrix products rather than explicit summations. For example, the Frobenius norm of a matrix can be written as
$\|\mathbf{A}\|_F = \sqrt{\mathrm{Tr}(\mathbf{A}\mathbf{A}^\top)}.$

Several properties of the trace are especially important in deep learning and optimization. The trace is invariant under transposition,
$\mathrm{Tr}(\mathbf{A}) = \mathrm{Tr}(\mathbf{A}^\top),$
and invariant under [cyclic permutation](https://en.wikipedia.org/wiki/Cyclic_permutation) of matrix products when dimensions are compatible:
$$
\mathrm{Tr}(\mathbf{A}\mathbf{B}\mathbf{C})
= \mathrm{Tr}(\mathbf{B}\mathbf{C}\mathbf{A})
= \mathrm{Tr}(\mathbf{C}\mathbf{A}\mathbf{B}).
$$
This cyclic property is essential in matrix calculus, as it allows expressions to be rearranged so that derivatives with respect to a given variable can be taken systematically.

In deep learning, the trace operator is used mainly "behind the scenes". It appears in derivations of gradients, Jacobians, and Hessians, and in the formulation of losses and regularization terms involving matrix products. Modern deep learning frameworks rely on these trace identities internally when implementing automatic differentiation and optimized linear algebra algorithms, even though the trace operator itself rarely appears in user code.

[^bilinear]: A function $f(\mathbf{x},\mathbf{y})$ is called **bilinear** if it is linear in each argument separately when the other argument is held fixed. For the dot product $f(\mathbf{x},\mathbf{y})=\mathbf{x}\cdot\mathbf{y}$, this means:
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

[^inverse]: So far, inverses were defined by left multiplication:
$\mathbf{A}^{-1}\mathbf{A}=\mathbf{I}.$
A right inverse satisfies
$\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}.$
For square matrices, left and right inverses coincide.

[^maxnorm]: The $L_\infty$ norm is also known as the **uniform norm**. This name comes from functional analysis: a sequence of functions $\{f_n\}$ converges to a function $f$ under the metric induced by the uniform norm iff $f_n$ converges to $f$ *uniformly*, meaning the maximum deviation $\sup_x |f_n(x)-f(x)|$ goes to zero. In finite-dimensional vector spaces, this reduces to taking the maximum absolute component.

[^unitvector]: A **unit vector** is a vector with unit Euclidean norm:
$
\|\mathbf{x}\|_2=1.
$


