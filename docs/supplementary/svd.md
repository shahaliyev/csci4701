# Singular Value Decomposition

<div style="margin:.3rem 0 1rem;font-size:.9em;color:#555;display:flex;align-items:center;gap:.35rem;font-family:monospace">
  <time datetime="2025-10-16">16 Oct 2025</time>
</div>

Presented below is an important concept of the introductory [linear algebra](../../mathematics/02_linear_algebra) called **singular value decomposition**, frequently implemented in machine/deep learning frameworks for optimizing calculations. It is the engine of [Principal Component Analysis](../pca) and is used for obtaining [pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse). Below, I tried to build the understanding gradually by providing necessary background and quick proofs. The text presumes prior knowledge on [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors).  

## Eigendecomposition
By definition, multiplying a matrix $A$ by one of its eigenvectors scales that eigenvector by the corresponding eigenvalue. Hence, $AQ = Q\Lambda$, where $Q$ is the matrix whose columns are the linearly independent eigenvectors of $A$ (if such a full set exists, $Q$ forms an **eigenbasis**), and $\Lambda$ is a diagonal matrix whose diagonal entries are the eigenvalues of $A$. If $Q$ is invertible, we can write: 
$$
A Q Q^{-1} = Q \Lambda Q^{-1} \quad \Rightarrow \quad A = Q \Lambda Q^{-1}.
$$
As a result, applying $A$ to any vector $v$ is equivalent to applying $Q \Lambda Q^{-1}$ to $v$, since both represent the same linear transformation. Geometrically, the coordinates of $v$ are first expressed in the eigenbasis via $Q^{-1}$, then scaled by the eigenvalues via $\Lambda$, and finally transformed back to the original standard basis (e.g. $\hat{\imath}, \hat{\jmath}$) via $Q$.

For a **symmetric matrix**, eigenvectors corresponding to distinct eigenvalues are orthogonal. If $A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2 \mathbf{v}_2$, then $ \mathbf{v}_1^T A \mathbf{v}_2 = \lambda_2 \mathbf{v}_1^T \mathbf{v}_2 $. Because $A^T = A$ and $(XY)^T = Y^T X^T$, we can write $ \mathbf{v}_1^T A \mathbf{v}_2 = (A^T \mathbf{v}_1)^T \mathbf{v}_2 = (A \mathbf{v}_1)^T \mathbf{v}_2 = \lambda_1 \mathbf{v}_1^T \mathbf{v}_2$. Combining both gives $(\lambda_1 - \lambda_2)\mathbf{v}_1^T \mathbf{v}_2 = 0$, and since $\lambda_1 \neq \lambda_2$, it follows that $\mathbf{v}_1^T \mathbf{v}_2 = 0 $.

Thus, the eigenvectors can be chosen **orthonormal**, and $Q$ is orthogonal with $Q^{-1} = Q^T$. The diagonalization becomes 
$$
A = Q \Lambda Q^T.
$$
Geometrically, orthogonal $Q$ represents a rotation or reflection, which preserves inner products and thus lengths and angles. This means orthogonal transformations preserve the geometric structure of space: 
$$ 
\| Q\mathbf{v} \|^2 = (Q\mathbf{v})^T (Q\mathbf{v}) = \mathbf{v}^T Q^T Q \mathbf{v} = \mathbf{v}^T \mathbf{v} = \| \mathbf{v} \|^2,
$$ 
showing that the squared length remains unchanged. Consequently, a symmetric matrix scales space only along mutually perpendicular eigenvector directions without introducing shear or non-uniform angular distortion.

## Singular Value Decomposition

*Eigendecomposition* described above is a powerful tool, but it has two major limitations: it only applies to square matrices, and even then, not every square matrix is diagonalizable (some lack a full set of linearly independent eigenvectors). In practical applications we usually deal with rectangular matrices or non-diagonalizable ones.

*Quadratic form* is an expression of the type $ \mathbf{x}^T A \mathbf{x} $, where $A$ is a square matrix and $\mathbf{x} \in \mathbb{R}^n$. It arises naturally when measuring the squared length of a vector after a linear transformation.
If we first transform $\mathbf{x}$ by a matrix $A$, the squared length becomes
$$
\|A\mathbf{x}\|^2 = (A\mathbf{x})^T (A\mathbf{x}) = \mathbf{x}^T A^T A \mathbf{x}.
$$
Quadratic form shows how $A$ scales or distorts space along different directions. Let $ \mathbf{y} = A \mathbf{x} $, then $\mathbf{x}^T A \mathbf{x} = \mathbf{x}^T (A \mathbf{x}) = \mathbf{x}^T \mathbf{y}$ is the dot product between the original vector $ \mathbf{x} $ and its transformed version $ A\mathbf{x} $. If $ \mathbf{x}^T A \mathbf{x} $ is large and positive, then $ A $ stretches $ \mathbf{x} $ strongly in roughly the same direction. If $ \mathbf{x}^T A \mathbf{x} = 0 $, then $ A $ sends $ \mathbf{x} $ to a vector orthogonal to itself -- a pure rotation-like behavior. If $ \mathbf{x}^T A \mathbf{x} $ is negative, then $ A $ flips $ \mathbf{x} $ in the opposite direction to its original orientation.

Both $A^T A$ (size $n \times n$) and $A A^T$ (size $m \times m$) are always symmetric. This is because $(A^T A)^T = A^T (A^T)^T = A^T A$, and similarly $(A A^T)^T = (A^T)^T A^T = A A^T$. They are also positive semidefinite: for any nonzero vector $\mathbf{x}$, we have quadratic form
$ \mathbf{x}^T (A^T A) \mathbf{x} = (A \mathbf{x})^T (A \mathbf{x}) = \| A \mathbf{x} \|^2 \geq 0 $, and  $ \mathbf{y}^T (A A^T) \mathbf{y} = (A^T \mathbf{y})^T (A^T \mathbf{y}) = \| A^T \mathbf{y} \|^2 \geq 0 $. Because they are symmetric, both matrices are diagonalizable with orthogonal eigenvectors. 

Let $A^T A \mathbf{v}_i = \sigma_i^2 \mathbf{v}_i$, where $\mathbf{v}_i$ are orthonormal eigenvectors and $\sigma_i^2$ are the eigenvalues of $A^T A$. Then:
$$
\|A \mathbf{v}_i\|^2 = \mathbf{v}_i^T A^T A \mathbf{v}_i = \sigma_i^2 \mathbf{v}_i^T \mathbf{v}_i = \sigma_i^2
$$
implies that $\sigma_i^2$ is the squared length of the transformed unit vector. Because $A^T A$ is positive semidefinite, we define $\sigma_i = \sqrt{\sigma_i^2} \ge 0$ as the *singular values* of $A$. The new vector $A\mathbf{v}_i$ has length $\sigma_i$, hence the matrix $A$ transforms the eigenvectors of $A^T A$ by scaling their lengths by the square roots of the eigenvalues (singular values) of $A^T A$. The eigenvectors themselves indicate the directions along which $A$ changes length by exactly $\sigma_i$.

We define the *Singular Value Decomposition (SVD)* of $A$ as:
$$
A = U \Sigma V^T.
$$
Here $V$ is an $n \times n$ orthogonal matrix whose columns $\mathbf{v}_i$ are eigenvectors of $A^T A$ (*right singular vectors*), $\Sigma$ is an $m \times n$ diagonal matrix whose diagonal entries $\sigma_i$ are the singular values of $A$, and $U$ is an $m \times m$ orthogonal matrix whose columns $\mathbf{u}_i$ are eigenvectors of $A A^T$ (*left singular vectors*).

The columns of $U$ arise similarly from the eigendecomposition of $A A^T$. If $A A^T \mathbf{u}_i = \sigma_i^2 \mathbf{u}_i,$ then $\mathbf{u}_i$ is a unit eigenvector of $A A^T$, and $\sigma_i^2$ is the same eigenvalue as before. Since $A A^T$ is $m \times m$ and symmetric, it is diagonalizable with an orthonormal set of eigenvectors forming the columns of $U$.

Once we have computed the eigenvalues and right singular vectors of $A^T A$, we can obtain the left singular vectors directly from them. Multiplying both sides of $A^T A \mathbf{v}_i = \sigma_i^2 \mathbf{v}_i$ by $A$ gives
$ (A A^T)(A \mathbf{v}_i) = \sigma_i^2 (A \mathbf{v}_i) $, 
so $A \mathbf{v}_i$ is an eigenvector of $A A^T$ with eigenvalue $\sigma_i^2$. To achieve orthonormal basis, we normalize $A \mathbf{v}_i$ to unit length. Having $\|A \mathbf{v}_i\| = \sigma_i$ leads to $\mathbf{u}_i = \sigma_i^{-1} A \mathbf{v}_i.$ Multiplying $ A \mathbf{v}_i = \sigma_i \mathbf{u}_i $ by $A^T$ on the left yields $ A^T \mathbf{u}_i = \sigma_i \mathbf{v}_i $. These equations guarantee that SVD formula is consistent: multiplying both sides on the right by $V$ gives $A V = U \Sigma$, and multiplying on the left by $U^T$ gives $U^T A = \Sigma V^T$.

Geometrically, $A$ takes the unit vector in the direction of $ \mathbf{v}_i $ in the input space, rotates (or reflects) it into the new orthogonal direction $ \mathbf{u}_i $ in the output space, and stretches it by a factor of $ \sigma_i $. Similarly, $A^T$ takes the unit vector in the direction of $ \mathbf{u}_i $ in the output space, maps it back into the corresponding direction $ \mathbf{v}_i $ in the input space, and scales it by the same factor $ \sigma_i $. Hence, $A$ can be understood as a composition of three transformations: a rotation or reflection by $V^T$, a scaling along orthogonal axes by $\Sigma$, and a rotation or reflection by $U$. For any vector $\mathbf{v}$,
$ A\mathbf{v} = U \Sigma V^T \mathbf{v} $, 
so $\mathbf{v}$ is first expressed in the basis of right singular vectors by $V^T$, then scaled by singular values by $\Sigma$, and finally mapped into the new space in the basis of left singular vectors by $U$. Thus, we extend the concept of eigendecomposition to all real matrices.