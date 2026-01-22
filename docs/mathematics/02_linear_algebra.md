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
v_1 \\
v_2 \\
\vdots \\
v_n
\end{bmatrix}
\in \mathbb{R}^n .
$$

In deep learning, such a vector is typically understood operationally: it is stored as a contiguous array of \(n\) real numbers and is mathematically an element of \(\mathbb{R}^n\), the Cartesian product of \(\mathbb{R}\) with itself \(n\) times. In this context, its "dimension" refers simply to its length \(n\). When \(n = 2\) or \(n = 3\), the vector can be visualized geometrically as a point or an arrow. When \(n\) is large, direct visualization is no longer possible, but the same algebraic operations—addition, scalar multiplication, dot products, and linear transformations—still apply. 

!!! note
    Depending on context, vectors can be visualized in different ways. In geometry and physics, they are often drawn as arrows representing magnitude and direction. In other settings, a vector can be viewed as a function that assigns a value to each index or coordinate. These visualizations are useful for building intuition, especially in low dimensions, but they do not alter the underlying algebraic definition of a vector. Linear algebra itself does not rely on geometric interpretation. It is fundamentally an algebraic theory of vector spaces and linear maps, and all definitions and results are independent of visualization. Geometry serves only as an intuitive aid not as a prerequisite. Beyond three dimensions, geometry in the literal, visual sense becomes unusable. Since most representations in deep learning live in very high-dimensional spaces, geometric visualization is generally not available and plays no direct role in practice. What remains meaningful are algebraic and analytical notions—such as inner products, norms, projections, and linear maps—rather than pictures or spatial intuition.

## Matrices and Tensors


A [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)) is a rectangular array of numbers arranged in rows and columns, satisfying properties of addition and multiplication (perhaps, from both deep learning and mathematics points of view). Formally, a real-valued matrix with \(m\) rows and \(n\) columns is an element of \(\mathbb{R}^{m \times n}\):

$$
\mathbf{A} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}.
$$

From a mathematical point of view, a matrix represents a [linear map](https://en.wikipedia.org/wiki/Linear_map) between vector spaces. Given a vector \(\mathbf{x} \in \mathbb{R}^n\), multiplication by a matrix \(\mathbf{A} \in \mathbb{R}^{m \times n}\) produces a new vector \(\mathbf{b} \in \mathbb{R}^m\): $$
\mathbf{A}\mathbf{x} = \mathbf{b}.$$ 

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

<!-- 

## 4) Transpose: swapping roles of rows and columns

The **transpose** flips a matrix across its diagonal:
\[
(A^\top)_{i,j} = A_{j,i}
\]
For vectors, transpose switches between column and row representations.

Why it matters:
- dot products are written as \(x^\top y\),
- many identities become simple with transpose,
- gradients often appear naturally as row vs column objects.

This is foundational notation in ML texts :contentReference[oaicite:2]{index=2}.

---

## 5) Basic operations: add, scale, and broadcast

### 5.1 Addition
Matrices (or vectors) can be added if they have the same shape:
\[
C = A + B \quad \Rightarrow \quad C_{i,j} = A_{i,j} + B_{i,j}
\]

### 5.2 Scalar multiplication (and affine shifts)
Scaling a matrix by a scalar:
\[
D = aB \quad \Rightarrow \quad D_{i,j} = aB_{i,j}
\]
You can also add a scalar to every entry:
\[
D = aB + c
\]

### 5.3 Broadcasting (common in DL)
In deep learning practice, we often add a vector to a matrix “by rows” without explicitly copying it:
\[
C = A + b \quad \Rightarrow \quad C_{i,j} = A_{i,j} + b_j
\]
This is broadcasting: implicit replication of \(b\) across rows :contentReference[oaicite:3]{index=3}.

!!! note
    Broadcasting is not “new math,” it’s a notational and computational convenience. But it changes how you think about shapes, which is why shape discipline is a real skill.

---

## 6) Matrix–vector and matrix–matrix multiplication

Multiplication is where linear algebra becomes a language of transformations.

### 6.1 Matrix–vector: linear combination of columns
If \(A \in \mathbb{R}^{m \times n}\) and \(x \in \mathbb{R}^n\), then:
\[
y = Ax \in \mathbb{R}^m
\]
Component form:
\[
y_i = \sum_{k=1}^n A_{i,k}x_k
\]
Interpretation: \(Ax\) is a weighted sum of the columns of \(A\):
\[
Ax = \sum_{j=1}^n x_j A_{:,j}
\]
This “weighted columns” view is essential when you interpret solutions of \(Ax=b\) :contentReference[oaicite:4]{index=4}.

### 6.2 Matrix–matrix: composing linear maps
If \(A \in \mathbb{R}^{m \times n}\) and \(B \in \mathbb{R}^{n \times p}\), then:
\[
C = AB \in \mathbb{R}^{m \times p}
\]
with
\[
C_{i,j} = \sum_{k=1}^n A_{i,k}B_{k,j}
\]
Interpretation: row \(i\) of \(A\) dotted with column \(j\) of \(B\).

### 6.3 Not commutative (but associative)
Matrix multiplication satisfies:
- distributive: \(A(B+C)=AB+AC\)
- associative: \(A(BC)=(AB)C\)
- generally **not commutative**: \(AB \neq BA\) :contentReference[oaicite:5]{index=5}

### 6.4 Elementwise (Hadamard) product is different
Elementwise multiplication \(A \odot B\) multiplies entries directly and is *not* the same as \(AB\) :contentReference[oaicite:6]{index=6}.

!!! note
    In DL code, both exist everywhere. Confusing them leads to wrong models and silent bugs.

---

## 7) Systems of linear equations: \(Ax=b\)

A compact equation:
\[
Ax=b
\]
represents \(m\) equations (rows), \(n\) unknowns (components of \(x\)) :contentReference[oaicite:7]{index=7}.

Expanded:
\[
A_{1,:}x=b_1,\;
A_{2,:}x=b_2,\;
\ldots,\;
A_{m,:}x=b_m
\]

Geometric interpretation:
- The columns of \(A\) are directions.
- The vector \(x\) tells you how much of each direction you combine.
- You reach \(b\) if and only if \(b\) lies in the **span** of those columns.

---

## 8) Identity and inverse (and why inversion is mostly theoretical)

### 8.1 Identity matrix
The identity matrix \(I_n\) satisfies:
\[
I_n x = x
\]
It has 1s on the diagonal and 0 elsewhere :contentReference[oaicite:8]{index=8}.

### 8.2 Inverse matrix
If \(A\) is invertible, its inverse \(A^{-1}\) satisfies:
\[
A^{-1}A = I
\]
Then the solution of \(Ax=b\) can be written:
\[
x = A^{-1}b
\]
But: computing or using \(A^{-1}\) directly is usually numerically worse than solving the system with methods that avoid explicit inversion :contentReference[oaicite:9]{index=9}.

!!! note
    In ML, “don’t invert; solve” is the standard rule. Inversion is conceptually helpful but computationally risky.

---

## 9) Linear dependence, span, and when solutions exist

### 9.1 Linear combination and span
A linear combination:
\[
\sum_i c_i v^{(i)}
\]
The **span** is the set of all such combinations.

### 9.2 Column space and solvability
For \(Ax=b\), the set of all achievable \(b\) is the **column space** of \(A\) (span of columns). The equation has a solution exactly when \(b\) lies in that span :contentReference[oaicite:10]{index=10}.

### 9.3 Linear independence
Vectors are **linearly independent** if none can be written as a linear combination of the others. Independence controls:
- whether the span has “full dimension,”
- whether solutions are unique.

### 9.4 Invertibility conditions
For \(A^{-1}\) to exist:
- \(A\) must be square (\(m=n\)),
- columns must be linearly independent.
A square matrix with dependent columns is **singular** :contentReference[oaicite:11]{index=11}.

---

## 10) Norms: measuring size (and why ML cares)

A **norm** is a function that measures “length” and satisfies three properties:
1. \( \|x\|=0 \Rightarrow x=0 \)
2. triangle inequality
3. scaling: \( \|\alpha x\| = |\alpha|\|x\| \) :contentReference[oaicite:12]{index=12}

### 10.1 \(L_p\) norms
\[
\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}, \quad p\ge 1
\]
Common cases:
- \(L_2\): Euclidean norm (default in many ML contexts)
- \(L_1\): encourages sparsity / “many zeros”
- \(L_\infty\): max magnitude component :contentReference[oaicite:13]{index=13}

### 10.2 Squared \(L_2\) norm
\[
\|x\|_2^2 = x^\top x
\]
Often easier for derivatives and computation :contentReference[oaicite:14]{index=14}.

### 10.3 Matrix norms (Frobenius)
\[
\|A\|_F = \sqrt{\sum_{i,j} A_{i,j}^2}
\]
Common in ML as the matrix analogue of Euclidean length :contentReference[oaicite:15]{index=15}.

---

## 11) Special vectors and matrices

These show up constantly in theory and practice.

### 11.1 Diagonal matrices
A diagonal matrix has nonzero entries only on the diagonal. Multiplying by it is cheap:
\[
\mathrm{diag}(v)x = v \odot x
\]
and inversion is cheap if all diagonal entries are nonzero :contentReference[oaicite:16]{index=16}.

### 11.2 Symmetric matrices
\[
A = A^\top
\]
Many important objects are symmetric (e.g., distance matrices, many Hessians under standard conditions) :contentReference[oaicite:17]{index=17}.

### 11.3 Unit vectors, orthogonality, orthonormality
- unit vector: \(\|x\|_2 = 1\)
- orthogonal: \(x^\top y = 0\)
- orthonormal: orthogonal + unit length :contentReference[oaicite:18]{index=18}

### 11.4 Orthogonal matrices
A matrix \(Q\) is orthogonal if:
\[
Q^\top Q = QQ^\top = I
\]
Then:
\[
Q^{-1} = Q^\top
\]
which makes many computations stable and cheap :contentReference[oaicite:19]{index=19}.

---

## 12) Eigendecomposition: directions a matrix scales

An eigenvector \(v\neq 0\) of a square matrix \(A\) satisfies:
\[
Av = \lambda v
\]
where \(\lambda\) is the eigenvalue :contentReference[oaicite:20]{index=20}.

Meaning:
- \(A\) transforms \(v\) by scaling (and possibly sign flip), not by changing direction.

### 12.1 Decomposition form
If \(A\) has a full set of independent eigenvectors:
\[
A = V\,\mathrm{diag}(\lambda)\,V^{-1}
\]
This reveals intrinsic structure of the transformation :contentReference[oaicite:21]{index=21}.

### 12.2 Symmetric case is especially clean
Every real symmetric matrix has:
\[
A = Q\Lambda Q^\top
\]
with \(Q\) orthogonal and \(\Lambda\) diagonal (real eigenvalues) :contentReference[oaicite:22]{index=22}.

### 12.3 Positive definiteness
Eigenvalues classify curvature-like behavior:
- **positive definite**: all eigenvalues \(>0\)
- **positive semidefinite**: all eigenvalues \(\ge 0\)

These guarantee:
\[
x^\top A x \ge 0
\]
(and stricter properties when definite) :contentReference[oaicite:23]{index=23}.

!!! note
    This is one of the cleanest bridges to optimization: curvature, stability, and “how stiff” a problem is show up as eigenvalues.

---

## 13) Singular Value Decomposition (SVD): the universal factorization

Eigenvalues require square matrices and can fail in general. SVD works for **every** real matrix :contentReference[oaicite:24]{index=24}.

For \(A \in \mathbb{R}^{m\times n}\):
\[
A = U D V^\top
\]
- \(U\): orthogonal (\(m\times m\))
- \(V\): orthogonal (\(n\times n\))
- \(D\): diagonal-shaped (\(m\times n\)), diagonal entries are singular values \(\sigma_i\) :contentReference[oaicite:25]{index=25}.

Interpretation (high-level):
- \(V^\top\): rotate/reflect input space
- \(D\): scale along principal axes (singular values)
- \(U\): rotate/reflect output space

Relationships:
- right singular vectors = eigenvectors of \(A^\top A\)
- left singular vectors = eigenvectors of \(AA^\top\)
- singular values relate to eigenvalues of these symmetric matrices :contentReference[oaicite:26]{index=26}.

---

## 14) Pseudoinverse: solving when inverse doesn’t exist

When \(A\) is not square or is singular, \(A^{-1}\) doesn’t exist. The Moore–Penrose pseudoinverse \(A^+\) generalizes “best possible inversion” :contentReference[oaicite:27]{index=27}.

Using SVD \(A = UDV^\top\):
\[
A^+ = V D^+ U^\top
\]
where \(D^+\) inverts nonzero singular values and transposes the diagonal-shaped matrix :contentReference[oaicite:28]{index=28}.

Two key regimes:

- **More columns than rows (underdetermined):** many solutions; pseudoinverse gives the one with minimal \(\|x\|_2\).
- **More rows than columns (overdetermined):** may be no exact solution; pseudoinverse gives the \(x\) minimizing \(\|Ax-b\|_2\).

This is the linear algebra behind least-squares behavior.

---

## 15) Trace: a compact way to write sums

The trace sums diagonal entries:
\[
\mathrm{Tr}(A)=\sum_i A_{i,i}
\]
It is useful because it interacts nicely with products and transpose, and it lets you rewrite expressions compactly (e.g., Frobenius norm) :contentReference[oaicite:29]{index=29}.

Example identity:
\[
\|A\|_F = \sqrt{\mathrm{Tr}(AA^\top)}
\]
And cyclic permutation property (when shapes allow):
\[
\mathrm{Tr}(ABC)=\mathrm{Tr}(BCA)=\mathrm{Tr}(CAB) -->
