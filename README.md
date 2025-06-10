# Optimal Design with the MaxVol Algorithm

This repository contains a simple implementation of MaxVol algorithms for optimal design,
using the D-optimality criterion.
The classical reference for this method is the book
"Theory of Optimal Experiments" V. V. Fedorov (1972), Academic Press.
For a short overview of this class of methods, see the Wikipedia article on
[Optimal Design of Experiments](https://en.wikipedia.org/wiki/Optimal_design_of_experiments).
The derivation below is based on the author's understanding of the algorithms,
but can probably be found in many other sources as well.

The goal of this repository is to provide a minimal self-contained derivation and implementation,
which is not intended to be the most efficient,
but rather to illustrate the basic principles of MaxVol algorithms.

## Licenses

Copyright 2025, Toon Verstraelen.

The tutorials in the Jupyter notebooks and the Markdown files are licensed under the
[Creative Commons CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) license.
The source code in the `optdesign` Python package is licensed under the
[LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html) license.

## Requirements

You can install the `optdesign` package from this repository
using the following command:

```bash
pip install optdesign
```

The Jupyter notebooks make use of the following Python packages:

- `numpy` <https://numpy.org/>
- `matplotlib` <https://matplotlib.org/>

These are readily available when opening the notebooks in Google Colab.
(See link below.)

## Theory

This section contains a quick recap of the theory of multivariate linear regression
and the D-optimality criterion.

### Multivariate Linear Regression

Consider the following linear set of equations:

$$
  \mathbf{y} = \mathbf{X} \hat{\mathbf{\beta}} + \hat{\mathbf{\epsilon}}
$$

where $\mathbf{y}$ is the vector of observations, $\mathbf{X}$ is the design matrix,
$\hat{\mathbf{\beta}}$ is the vector of coefficients, and $\hat{\mathbf{\epsilon}}
$
is the vector of errors.
All stochastic quantities are denoted with hats.
We will make the typical assumption that the errors are independent
and identically distributed (i.i.d.) with mean zero and variance $\sigma^2$.

The solution of this system of equations is given by:

$$
  \hat{\mathbf{\beta}}
  = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top (\mathbf{y} - \hat{\mathbf{\epsilon}})
$$

The expected value of the parameter vector $\mathbf{\beta}$ is then simply given by:

$$
  \mathbf{\beta}
  = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

where we use the notation $E[\hat{\mathbf{\beta}}] = \mathbf{\beta}$.
The covariance matrix of the parameter vector $\mathbf{\beta}$ is given by:

$$
  \operatorname{COV}[\hat{\beta}_i, \hat{\beta}_j]
  = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}_{ij}
$$

The covariance matrix is thus known before the experiment is performed,
i.e. before $\mathbf{y}$ is observed.
It is already fixed when making the decision of where or what to measure,
i.e. when choosing the design matrix $\mathbf{X}$.

### $D$-Optimality Criterion

The goal of the $D$-optimality criterion is to choose the design matrix $\mathbf{X}$
such that the determinant of the covariance matrix is minimized.
In other words, we want to maximize the following determinant:

$$
  |\mathbf{X}^\top \mathbf{X}|
$$

This can be seen as a measure of the "volume" of the
[information matrix](https://en.wikipedia.org/wiki/Fisher_information)
$\mathbf{X}^\top \mathbf{X}$, hence the name "MaxVol" algorithm.

To facilitate manipulations of the design matrix $\mathbf{X}$,
we will make use of the
[Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
(SVD) of the design matrix $\mathbf{X}$:

$$
  \mathbf{X} = \mathbf{U} \mathbf{s} \mathbf{V}^\top
$$

where $\mathbf{U}$ and $\mathbf{V}^\top$ have orthogonal columns,
and $\mathbf{s}$ is a diagonal matrix with the singular values.
When there are more equations than unknowns, i.e. for a typical regression problem,
the matrix $\mathbf{U}$ has the same shape as $\mathbf{X}$,
and the matrix $\mathbf{V}$ is a square matrix.

The determinant of the information matrix can then be expressed as:

$$
  |\mathbf{X}^\top \mathbf{X}| = |\mathbf{s}^2| = \prod_{n=1}^N s_n^2
$$

where $N$ is the number of singular values (and unknowns).
If $\mathbf{X}$ is singular, the volume is obviously zero.

### MaxVol Algorithm: Row Addition and Row Removal

Consider an initial non-singular design matrix $\mathbf{X}$,
which is a $M \times N$ matrix with $M$ measurements and $N (\le M)$ unknowns.
We also have a proposal for a new measurement $\mathbf{a}$,
in the form of a new vector that should be included in the design matrix.
Here we will show how to compute the change in determinant after this addition.

Consider the following addition of a row vector $\mathbf{a}^\top$ to the design matrix $\mathbf{X}$:

$$
  \mathbf{X}^\text{add}
  = \begin{bmatrix} \mathbf{X} \\\hline \\[-0.9em] \mathbf{a}^\top \end{bmatrix}
$$

We work out the determinant after row addition
by making use of the SVD of the initial design matrix
$\mathbf{X} = \mathbf{U} \mathbf{s} \mathbf{V}^\top$:

$$
  \left|\left(\mathbf{X}^\text{add}\right)^\top \mathbf{X}^\text{add}\right|
  = \left|\mathbf{V}^\top \mathbf{s}^2 \mathbf{V} + \mathbf{a} \mathbf{a}^\top\right|
$$

The second term is a rank-1 matrix.
We now make use of the following properties of matrix determinants:

- The determinant is unvariant under orthogonal transformations of the rows and columns.
- The determinant of a product of matrices is the product of the determinants.

Left-multiplying with $\mathbf{V}$ and right-multiplying with $\mathbf{V}^\top$ yields:

$$
  \left|\left(\mathbf{X}^\text{add}\right)^\top \mathbf{X}^\text{add}\right|
  = \left|\mathbf{s}^2 + \mathbf{V}  \mathbf{a} \mathbf{a}^\top \mathbf{V}^\top \right|
$$

Then we factor out the singular values:

$$
  \left|\left(\mathbf{X}^\text{add}\right)^\top \mathbf{X}^\text{add}\right|
  = |\mathbf{s}|^2
    \left|
        \mathbf{I} +
        \mathbf{s}^{-1} \mathbf{V} \mathbf{a} \mathbf{a}^\top \mathbf{V}^\top \mathbf{s}^{-1}
    \right|
$$

For the last step, we introduce $\mathbf{u} = \mathbf{s}^{-1} \mathbf{V}  \mathbf{a}$,
to clarify that the first factor is the determinant of an identity matrix plus a rank-1 update:

$$
    \left|\left(\mathbf{X}^\text{add}\right)^\top \mathbf{X}^\text{add}\right|
    = |\mathbf{s}|^2 \left| \mathbf{I} + \mathbf{u} \mathbf{u}^\top \right|
$$

The matrix in the second determinant is diagonal in any basis
where $\mathbf{u}$ is one of the basis vectors,
with corresponding eigenvalue $1 + \|\mathbf{u}\|^2$.
All other eigenvalues are equal to $1$.
Hence, the determinant can be computed as:

$$
  \left|\left(\mathbf{X}^\text{add}\right)^\top \mathbf{X}^\text{add}\right|
  = \left|\mathbf{X}^\top \mathbf{X}\right| (1 + \|\mathbf{u}\|^2)
$$

The second factor expresses the change in determinant
after the addition of the new vector $\mathbf{a}$.
One can rewrite this in a more convenient form:

$$
  1 + \|\mathbf{u}\|^2
  = 1 + \mathbf{a}^\top \mathbf{V} \mathbf{s}^{-2}\, \mathbf{V}^\top \mathbf{a}
$$

When testing a large set of candidate vectors $\mathbf{a}$,
one performs the SVD of the design matrix $\mathbf{X}$ only once,
and then computes the norms of $\mathbf{s}^{-1} \mathbf{V}  \mathbf{a}$ for each candidate vector $\mathbf{a}$.
The largest norm corresponds to the vector that maximizes the determinant after addition.
The same expression can be used to determine which row to remove
(which will always reduce the volume) as to get the smallest reduction in volume.
The function `opt_dmetric()` in the `optdesign` packages uses this approach
to construct an optimal design matrix in a "greedy" fashion,
i.e. by always adding and removing the optimal row at each step.
Note that such a greedy algorithm performs a local search
and does not guarantee a globally optimal design matrix.

### MaxVol Algorithm: Row Replacement

Consider a square matrix $\mathbf{X}$ with $M = N$ rows and columns.
For this case, one can also merge the addition and removal of a row
into a single replacement operation
and directly compute the change in volume due to the replacement.

The derivation of this algorithm makes use of Cramer's rule,
which can be used to express the inverse of a matrix as:

$$
  \mathbf{X}^{-1} = \frac{1}{|\mathbf{X}|} \operatorname{adj}(\mathbf{X})
$$

where $\operatorname{adj}(\mathbf{X})$ is the adjugate of the matrix $\mathbf{X}$,
i.e. the transpose matrix of cofactors.
Each cofactor is the determinant of the matrix
after removing the corresponding row and column from $\mathbf{X}$.

We will derive an expression for the determinant of a matrix $\mathbf{X}'$,
which is obtained by replacing a row $k$ in $\mathbf{X}$
with a new vector $\mathbf{r}^\top$.
The determinant of the new matrix $\mathbf{X}'$ can be expressed as:

$$
  |\mathbf{X}'| \mathbf{I}
  = \mathbf{X}' \operatorname{adj}(\mathbf{X'})
$$

Of this matrix identify, we need only the diagonal element with indexes $(k, k)$:

$$
  |\mathbf{X}'|
  = \sum_{\ell=1}^N
    \mathbf{r}_{\ell} \Bigl[\operatorname{adj}(\mathbf{X'})\Bigr]_{\ell k}

$$

Because $\mathbf{X}$ and $\mathbf{X}'$ differ only in row $k$,
the column they have the matrix elements $\Bigl[\operatorname{adj}(\mathbf{X'})\Bigr]_{\ell k}$
for all $\ell \in \{1, \ldots, N\}$. Hence, we can replace the adjugate matrix,
and rewrite it in terms of $\mathbf{X}^{-1}$ and $|\mathbf{X}|$:

$$
  |\mathbf{X}'|
  = \sum_{\ell=1}^N
    \mathbf{r}_{\ell} \Bigl[\operatorname{adj}(\mathbf{X})\Bigr]_{\ell k}
  = |\mathbf{X}| \sum_{\ell=1}^N
    \mathbf{r}_{\ell} \Bigl[\mathbf{X}^{-1}\Bigr]_{\ell k}
$$

In conclusion, the change in determinant after replacing a row $k$ in $\mathbf{X}$
with a new vector $\mathbf{r}^\top$ is given by the following simple expression:

$$
  \frac{|\mathbf{X}'|}{|\mathbf{X}|}
  = (\mathcal{r} \mathbf{X}^{-1})_k
$$

If one has a matrix $\mathbf{C}$ whose rows are all candidate vectors $\mathbf{r}^\top$,
one can compute the change in determinant for all combinations of candidates and rows to replace
by computing the matrix product $\mathbf{C} \mathbf{X}^{-1}$.
The largest absolute value in the resulting matrix corresponds to the row to be replaced
and candidate put in its place.
Note that $\mathbf{C}$ can be rectangular, i.e. it can have more rows than columns.

The function `opt_maxvol()` in the `optdesign` package uses this approach to construct
a square optimal design matrix in a "greedy" fashion,
i.e. by always replacing the optimal row at each step.
Again, this corresponds to a local search and does not guarantee a globally optimal design matrix.

### Other utilities in the `optdesign` package

The `optdesign` package also contains a few other utilities to set up a design matrix.
The function `setup_greedy()` will start building a square matrix from a set of candidate vectors,
to initiate the greedy algorithm.
It does this by selecting each row to be added to maximize
the non-zero singular values of the design matrix being built.
The function `setup_random()` will randomly select a set of rows from the candidate vectors.
