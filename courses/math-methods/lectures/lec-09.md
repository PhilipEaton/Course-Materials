---
layout: default
title: Mathematical Methods - Lecture 09
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 9
---



# Lecture 09 – The Eigenvalue Problem and Eigenvectors


## Eigenvalue Problem: Finding Eigenvalues — Quick Review

In this lecture, we continue our exploration of one of the most important problems in linear algebra: the **eigenvalue problem**:

$$
\mathbf{A} \vec{v} = \lambda \vec{v}
$$

This equation captures a fundamental idea: When a matrix $\mathbf{A}$ acts on a vector $\vec{v}$, the result is the *same vector direction*, scaled by a factor $\lambda$. The vector may flip direction if $\lambda$ is negative, but this is a reflection and not a rotation. The scalar $\lambda$ is called the **eigenvalue**, and the vector $\vec{v}$ is called the **eigenvector**. Together, they describe how $\mathbf{A}$ acts on special directions in space.

If $\mathbf{A}$ is an $n \times n$ matrix, there will generally be $n$ eigenvalues. Some of these eigenvalues may be repeated. This repetition is called **degeneracy** and indicates that multiple independent eigenvectors share the same eigenvalue. The total number of eigenvalues always matches the dimension of the space, although some eigenvalues may be complex or degenerate, which is perfectly acceptable.

Recall that we find eigenvalues by rewriting the eigenvalue equation as:

$$
\left( \mathbf{A} - \lambda \mathbf{I} \right)\vec{v} = \vec{0}
$$

and demanding a **non‑trivial solution** $\vec{v} \neq \vec{0}$. For such a solution to exist, the matrix $(\mathbf{A} - \lambda \mathbf{I})$ must be **singular**, meaning its determinant is zero:

$$
\det\!\left( \mathbf{A} - \lambda \mathbf{I} \right) = 0
$$

This equation is called the **characteristic equation**. Solving it yields the eigenvalues $\lambda$ of the matrix. For an $n \times n$ matrix, the characteristic equation is a polynomial of degree $n$.

Let us now deepen our understanding of what eigenvalues tell us about matrices and the systems they represent. After that we will move on to how to find eigenvectors once you have the eigenvalues.






## Eigenvalues and Matrix Properties

When solving the eigenvalue problem, it is useful to understand how basic matrix properties influence the behavior of eigenvalues. While eigenvalues depend on the specific matrix, they often encode important global information about the transformation the matrix represents.

### Determinant and the Product of Eigenvalues

A key relationship involves the determinant of $\mathbf{A}$. The determinant is equal to the product of the eigenvalues:

$$
\det(\mathbf{A}) = \lambda_1 \lambda_2 \cdots \lambda_n
$$

where $\lambda_1, \lambda_2, \ldots, \lambda_n$ are the eigenvalues of $\mathbf{A}$. We will prove this result in the next lecture.

This relationship provides immediate insight. If the determinant is zero, then at least one eigenvalue must be zero. Geometrically, this signals a loss of dimensionality, such as a projection onto a lower‑dimensional subspace.

### Trace and the Sum of Eigenvalues

Similarly, the trace of a matrix equals the sum of its eigenvalues:

$$
\mathrm{Tr}(\mathbf{A}) = \lambda_1 + \lambda_2 + \cdots + \lambda_n
$$

The trace is defined as the sum of the diagonal elements of $\mathbf{A}$ and will also be proven in the next lecture. In physics, the trace appears in many contexts, including expectation values in quantum mechanics and stability criteria in dynamical systems.


### Symmetric Matrices: Real Eigenvalues

A particularly important case occurs when $\mathbf{A}$ is a **real symmetric matrix**, meaning

$$
\mathbf{A} = \mathbf{A}^\text{T}
$$

and contains onlt real elements.

In this case, all eigenvalues of $\mathbf{A}$ will be real. This result has deep physical significance.

To see why, suppose $\vec{v}$ is an eigenvector of $\mathbf{A}$ with eigenvalue $\lambda$, where $\vec{v}$ and $\lambda$ are allowed to be complex. We begin with the eigenvalue problem:

$$
\mathbf{A}\vec{v} = \lambda \vec{v}
$$

Multiply on the left by $\vec{v}^\dagger$:

$$
\begin{aligned}
\mathbf{A} \vec{v} &= \lambda \vec{v} \\
\vec{v}^\dagger \mathbf{A} \vec{v} &= \vec{v}^\dagger  \lambda \vec{v} \\
\big( \mathbf{A}^\dagger \vec{v}\big)^\dagger \vec{v} &=  \lambda \vec{v}^\dagger  \vec{v}\\
\big( (\mathbf{A}^*)^\text{T} \vec{v}\big)^\dagger \vec{v} &=  \lambda \vec{v}^\dagger  \vec{v} \\
\end{aligned}
$$

but $ \mathbf{A} $ is real and symmetric, meaning $\mathbf{A}^* = \mathbf{A}$ and $\mathbf{A}^\text{T} = \mathbf{A}$ so we have:

$$
\begin{aligned}
\big( \mathbf{A} \vec{v}\big)^\dagger \vec{v} &=  \lambda \vec{v}^\dagger  \vec{v} \\
\big( \lambda \vec{v}\big)^\dagger \vec{v} &=  \lambda \vec{v}^\dagger  \vec{v} \\
\lambda^* \vec{v}^\dagger \vec{v} &=  \lambda \vec{v}^\dagger  \vec{v} \\
\lambda^* &=  \lambda 
\end{aligned}
$$

which is only possible if $\lambda$ is a real number.

This result explains why symmetric matrices arise so frequently in physics. Quantities such as inertia tensors, stiffness matrices, and Hamiltonians must have real eigenvalues because they correspond to measurable physical quantities.


### Orthogonal Matrices: Eigenvalues of Magnitude One

Orthogonal matrices introduce another important constraint. A matrix $\mathbf{A}$ is orthogonal if

$$
\mathbf{A}^{-1} = \mathbf{A}^\text{T}
$$

For orthogonal matrices, **all eigenvalues have magnitude one**. These eigenvalues may be real ($+1$ or $-1$) or complex, but they always satisfy

$$
\vert\lambda\vert = 1
$$

This property reflects the fact that orthogonal transformations preserve lengths and angles. In physics, this connects directly to rotations and reflections. In quantum mechanics, the complex analog of orthogonal matrices are **unitary matrices**, which preserve probabilities.

{% capture ex %}

As a concrete example, consider the two‑dimensional rotation matrix

$$
\mathbf{R}(\theta) =
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

We compute the eigenvalues by solving


$$
\begin{aligned}
\det(\mathbf{R} - \lambda \mathbf{I}) &= 0 \\[1.15ex]
\begin{vmatrix}
	\cos(\theta) - \lambda & -\sin(\theta) \\
	\sin(\theta) & \cos(\theta)- \lambda
\end{vmatrix} &= 0 \\[2.25ex]
\big(\cos(\theta) - \lambda\big)^2 - \big(-\sin^2(\theta)\big)  &= 0 \\[1.15ex]
\big(\cos(\theta) - \lambda\big)^2  &= - \sin^2(\theta) \\[1.15ex]
\cos(\theta) - \lambda  &= \pm i \sin(\theta) \\[1.15ex]
\lambda  &= \cos(\theta)  \pm i \sin(\theta) \\[1.15ex]
\end{aligned}
$$

Taking the magnitude gives

$$
|\lambda|^2 = \cos^2(\theta) + \sin^2(\theta) = 1
$$


The eigenvalues have unit magnitude, as prominsed for an orthogonal matrix.

**Side note:** You may recognise this result as the agular part of a complex number written in polar form:

$$
z = r \big( \cos(\theta)  + i \sin(\theta)\big)
$$

where $\theta$ is measured counterclockwise. If the angle were in the clockwsie direction we could take $\theta \rightarrow -\theta$ to get:

$$
z = r \big( \cos(-\theta)  + i \sin(-\theta)\big) \quad\Rightarrow\quad z = r \big( \cos(\theta)  - i \sin(\theta)\big)
$$

where we have used the fact that $\cos$ is an even function and $\sin$ is an odd function to simplify. 

You may also recall that these combination of angles and the complex unit $i$ was also found to be identical to the 2-dimentional rotation matrix. So, the eigenvalues of this matrix are represent rotations in the counterclockwise and clockwise directions, and no change in the magnitude of the vectors. 

Further, tt turns out if you expand for small angles $\theta \ll 1$ you can show that (this is something we explicity work out in Mathematical Physics, in case you are interested):

$$ 
e^{\pm i\theta} = \cos(\theta)  \pm i \sin(\theta)
$$

wich is called the Euler identity. More on this in Mathematical Physics. So, the eigenvalues for this matrix can be written as $e^{\pm i \theta}$. 

{% endcapture %}
{% include example.html content=ex %}




	







## Connection Between Eigenvalues and Eigenvectors

Now that we have discussed how to find the eigenvalues of a matrix and some of the properties of matrices they relate to, it’s time to connect these to eigenvectors, the other half of the eigenvalue problem. Recall, eigenvalues $\lambda$ tell us how much a matrix $\mathbf{A}$ scales a vector $\vec{v}$, without rotating it, through the equation:

$$
\mathbf{A} \vec{v} = \lambda \vec{v}
$$

But how do we determine the eigenvector $\vec{v}$? And why is $\vec{v}$ important?




### The Role of Eigenvectors

We may find ourselves asking the following question: Why do we care about eigenvectors? 

The short (and admittedly boring) answer is that they simplify how we understand and compute with matrices. This is something we will work with in the next lecture.

The longer (and far more interesting) answer is that eigenvectors reveal something fundamental about the system or operator the matrix represents. In physics, eigenvectors often correspond to:

- **Natural Modes:** Directions along which a system naturally oscillates, such as the normal modes of a vibrating molecule or bridge.  
- **Principal Axes:** Symmetry axes in systems like the moment of inertia tensor in rotational dynamics.  
- **Quantum States:** Physically meaningful states like energy eigenstates, where measurement yields definite outcomes.

In essence, **eigenvectors help us untangle complex systems by identifying their simplest, most natural directions of behavior**.

For a matrix operator $\mathbf{A}$, the eigenvectors define the **axes** along which $\mathbf{A}$ acts by simple scaling. Each eigenvector points along a direction that is preserved (not rotated or skewed), and its associated eigenvalue tells us how much vectors in that direction are stretched or compressed.

- If $\lambda > 0$, the vector is scaled (stretched or squished) along this axis.
- If $\lambda < 0$, the vector is scaled and reflected along this axis.
- If $\lambda = 0$, the transformation collapses that direction entirely.

When you apply $\mathbf{A}$ to a general vector, it typically scrambles it by rotating, skewing, and scaling the components in some unintuitive way. But eigenvectors cut through the chaos: they remain aligned with themselves. That kind of geometric clarity is powerful when trying to build a conceptual picture of what's happening inside a system, especially when the system is large, messy, or otherwise opaque.



 



 ### Solving for Eigenvectors

To find the eigenvectors corresponding to a given eigenvalue, we need to return to the eigenvalue equation:

$$
\mathbf{A} \vec{v} = \lambda \vec{v}
$$

and rewrite it in a more familiar form:

$$
(\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \vec{0}
$$

where $\mathbf{I}$ is the identity matrix. This tells us that $\vec{v}$ lies in the **null space** of the matrix $(\mathbf{A} - \lambda \mathbf{I})$. In other words, this matrix transforms $\vec{v}$ into the zero vector, so we’re looking for the vectors that get annihilated under this transformation. However, we’re only interested in **nontrivial** solutions here, which menas we want eigenvectors such that $\vec{v} \ne \vec{0}$.

This setup is important for deeper mathematical reasons, but practically speaking, this gives us a system of linear equations. Solving this system will give us the eigenvectors associated with the chosen eigenvalue $\lambda$.

It’s important to note that this equation will look different for each eigenvalue. That is, every distinct eigenvalue $\lambda$ you find from the characteristic equation will yield its own unique system of equations to solve.

Here’s a step-by-step roadmap for finding eigenvectors:

1. **Find the eigenvalues** by solving the characteristic equation $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$.
2. For each eigenvalue $\lambda$, **construct the matrix** $\mathbf{A} - \lambda \mathbf{I}$.
3. **Solve the homogeneous system** $(\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \vec{0}$ to find the eigenvectors.
    - If an eigenvalue is repeated (degenerate), make sure to find all **linearly independent** eigenvectors associated with it.
4. **Choose a scale** (or normalization) for each eigenvector.
    - We'll explain what this means and why it matters shortly.

This process is algorithmic and mechanical, but it opens the door to powerful geometric and physical interpretations once we have the eigenvectors in hand.





{% capture ex %}

Let’s solidify these steps with an example; it is honestly the best way to learn how to do this. 

Consider the matrix:

$$
\mathbf{A} = \begin{bmatrix}
4 & 1 \\
2 & 3
\end{bmatrix}
$$

#### Step 1: Find the eigenvalues

To find the eigenvalues we need to first find the eigenvalues. To do so we will set up and solve the characteristic equation:

$$ \text{det}( \mathbf{A} - \lambda \mathbf{I} ) = 0 $$

First, let's set up the matrix $( \mathbf{A} - \lambda \mathbf{I} )$:

$$ ( \mathbf{A} - \lambda \mathbf{I} ) = \begin{bmatrix}
4 & 1 \\
2 & 3
\end{bmatrix} - \lambda \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} = \begin{bmatrix}
4 - \lambda & 1 \\
2 & 3 - \lambda
\end{bmatrix} $$

From this, the characteristic equation would be:

$$ \text{det}( \mathbf{A} - \lambda \mathbf{I} ) = (4 - \lambda) (3 - \lambda) - 2  = 0 $$
and solving for $\lambda$:

$$
\begin{aligned}
(4 - \lambda) (3 - \lambda) - 2  &= 0  \\
12 - 7 \lambda +  \lambda^2 - 2  &= 0 \\
	\lambda^2 - 7 \lambda + 10   &= 0 \\
	(\lambda - 2 )(\lambda - 5)  &= 0 
\end{aligned}
$$

We have $\lambda =2$ or $\lambda = 5$ as our eigenvalues for this matrix. It would be helpful to give them names as we go to find their associated eigenvectors. Let's call them:

$$ 
\lambda_1 =2 \qquad \text{and} \qquad \lambda_2 = 5 
$$

the eigenvalues of $\mathbf{A}$. 

#### Step 2: Form $(\mathbf{A} - \lambda \mathbf{I})$ for each $\lambda$

This step only requires the substitution of the eigenvalues into $(\mathbf{A} - \lambda \mathbf{I})$. 

<div class="two-column">

<div class="column">

Substituting $\lambda_1 = 2$, we compute:<br> 

$$
\mathbf{A} - \lambda_1 \mathbf{I} = \begin{bmatrix}
	4 - 2 & 1 \\
	2 & 3 - 2
\end{bmatrix} = \begin{bmatrix}
	2 & 1 \\
	2 & 1
\end{bmatrix}
$$

</div>

<div class="column">

Substituting $\lambda_2 = 5$, we compute:<br>

$$
\mathbf{A} - \lambda_2 \mathbf{I} = \begin{bmatrix}
	4 - 5 & 1 \\
	2 & 3 - 5
\end{bmatrix} = \begin{bmatrix}
	-1 & 1 \\
	2 & -2
\end{bmatrix}
$$

</div>

</div>



#### Step 3: Solve $(\mathbf{A} - \lambda \mathbf{I}) \vec{v} = 0$ for each eigenvector
This step requires us to actually do some algebra and solve for the components of the eigenvectors. We will do both side-by-side since the steps are identical.

<div class="two-column">

<div class="column">

For $\vec{v}_1$ the equation to solve is:<br>

$$
\begin{bmatrix}
	2 & 1 \\
	2 & 1
\end{bmatrix}
\begin{bmatrix}
	a \\
	b
\end{bmatrix} = \begin{bmatrix}
	0 \\
	0
\end{bmatrix}
$$

Expanding, we get:<br>

$$
\text{Row 1: } 2a + b = 0, \qquad \text{Row 2: } 2a + b = 0 
$$

From Row 1, we can see that $b = -2a$. (Notice Row 2 gives the exact same result!)<br>

Thus, the general solution is:<br>

$$
\vec{v}_1 = \begin{bmatrix}
	a \\
	-2a
\end{bmatrix}
$$

where $a$ will be determined in the next step.

</div>

<div class="column">

For $\vec{v}_2$ the equation to solve is:<br>

$$
\begin{bmatrix}
	-1 & 1 \\
	2 & -2
\end{bmatrix}
\begin{bmatrix}
	c \\
	d
\end{bmatrix} = \begin{bmatrix}
	0 \\
	0
\end{bmatrix}
$$

Expanding, we get:<br>

$$
\text{Row 1: } -c + d = 0, \qquad \text{Row 2: } 2c - 2d = 0 
$$

From Row 1, we can see that $d = c$. (Notice Row 2 gives the exact same result!)<br>

Thus, the general solution is:<br>

$$
\vec{v}_2 = \begin{bmatrix}
	c \\
	c
\end{bmatrix}
$$

where $c$ will be determined in the next step.

</div>

</div>



Hopefully you noticed that the equations from Row 1 and Row 2 gave the same result for the unknown components of the eigenvector. That’s not a mistake, it’s expected and will always happen when solving for eigenvectors!

Why? Because when the determinant of a matrix is zero, at least one row is *linearly dependent* on the others. In simpler terms, one or more rows are *redundant* and doesn't provide any new information. For example:

- In a $2 \times 2$ matrix, if the determinant is zero, one row must be redundant.
- In a $3 \times 3$ matrix, if the determinant is zero, one or even two rows will be redundant.

This is why when we solve the system $(\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \vec{0}$, we always end up with fewer independent equations than unknowns. We will always end us with at least one unknown we cannot explicitly solve for.




#### Step 4: Select the Scale Parameter for the Eigenvector

The undetermined quantities, $a$ or $c$, are called **scale factors** for the eigenvector. Since eigenvectors represent the principal axes (i.e., directions) of a matrix, their exact length doesn’t really matter. We're primarily interested in their direction.

There is one notable exception: in quantum mechanics, eigenvectors are usually normalized, meaning their length is scaled to 1. This is important because the components of a quantum state vector are tied to probabilities, and probabilities must add up to 1.

In most other contexts, though, we simply pick the simplest value for the scale factor. Typically, this means choosing something that avoids fractions and keeps the math clean.

<div class="two-column">

<div class="column">

To simplify, we can take $a = 1$, giving:<br>

$$
\vec{v}_1 = \begin{bmatrix}
	1 \\
	-2
\end{bmatrix}
$$

This is the eigenvector associated with $\lambda_1 = 2$.<br>

</div>

<div class="column">

To simplify, we can take $c = 1$, giving:<br>

$$
\vec{v}_2 = \begin{bmatrix}
	1 \\
	1
\end{bmatrix}
$$

This is the eigenvector associated with $\lambda_2 = 5$.<br>

</div>

</div>




#### Interpreting the Result

The eigenvector 

$$\vec{v}_1 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$ 

represents a direction in the plane along which the transformation by $\mathbf{A}$ acts as simple scaling by $\lambda_1 = 2$. 

Similarly, the eigenvector 

$$\vec{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$ 

represents a direction in the plane along which the transformation by $\mathbf{A}$ acts as simple scaling by $\lambda_2 = 5$

{% endcapture %}
{% include example.html content=ex %}





### More Dimensions, More Complexity

For higher-dimensional matrices, the process is the same, it is just a bit more work. Let's look at an example of a $3\times 3$ matrix with a degenerate (i.e., repeated) eigenvalue.

{% capture ex %}

Consider the matrix:

$$
\mathbf{A} = \begin{bmatrix}
	2 & 0 & 0 \\
	0 & 3 & 1 \\
	0 & 1 & 3
\end{bmatrix}
$$

#### Step 1: Find the eigenvalues

To find the eigenvalues we need to set up and solve the characteristic equation:

$$ 
\text{det}( \mathbf{A} - \lambda \mathbf{I} ) = 0 
$$

First, let's set up the matrix $( \mathbf{A} - \lambda \mathbf{I} )$:

$$ ( \mathbf{A} - \lambda \mathbf{I} ) = \begin{bmatrix}
	2 & 0 & 0 \\
	0 & 3 & 1 \\
	0 & 1 & 3
\end{bmatrix} - \lambda \begin{bmatrix}
	1 & 0 & 0  \\
	0 & 1 & 0 \\
	0 & 0 & 1
\end{bmatrix} =  \begin{bmatrix}
2 - \lambda & 0 & 0 \\
0 & 3 - \lambda & 1 \\
0 & 1 & 3 - \lambda
\end{bmatrix} $$

From this, the characteristic equation would be:

$$ \text{det}( \mathbf{A} - \lambda \mathbf{I} ) = (2 - \lambda) \begin{vmatrix}
	3 - \lambda & 1 \\
	1 & 3 - \lambda
\end{vmatrix} =  (2 - \lambda) \left( \frac{}{}   (3 - \lambda)^2 - 1   \frac{}{} \right) = 0 
$$

where we have used the cofactor method to calculate the determinant. Now, solving for $\lambda$:

$$
\begin{aligned}
	(2 - \lambda) \left( \frac{}{}   (3 - \lambda)^2 - 1   \frac{}{} \right) &= 0  \\
	(2 - \lambda) \left( \frac{}{}   (9 - 6 \lambda + \lambda^2) - 1   \frac{}{} \right) &= 0  \\
	(2 - \lambda) \left( \frac{}{}   \lambda^2 - 6 \lambda + 8   \frac{}{} \right) &= 0  \\
	(2 - \lambda)  (\lambda - 4)( \lambda - 2 )    &= 0 
\end{aligned}
$$

meaning $\lambda =2$, $\lambda = 4$, or $\lambda = 2$ are our eigenvalues. Notice there is a degeneracy here, $\lambda = 2$ happened twice! That means there are two eigenvectors with eigenvalues of $2$. 

Now that we are going to find their eigenvectors, we need to give these names. Let's call them:

$$ 
\lambda_1 = 4 \qquad \text{and} \qquad \lambda_2 = \lambda_3 = 2 
$$

the eigenvalues of $\mathbf{A}$. 

#### Step 2: Form $(\mathbf{A} - \lambda \mathbf{I})$ for each $\lambda$
This step only requires the substitution of the eigenvalues into $(\mathbf{A} - \lambda \mathbf{I})$. 

Substituting $\lambda_1 = 4$, we compute:

$$
\mathbf{A} - \lambda_1 \mathbf{I} = \begin{bmatrix}
	2 - 4 & 0 & 0 \\
	0 & 3 - 4 & 1 \\
	0 & 1 & 3 - 4
\end{bmatrix} = \begin{bmatrix}
-2 & 0 & 0 \\
0 & -1 & 1 \\
0 & 1 & -1
\end{bmatrix}
$$


Substituting $\lambda_2 = \lambda_3 = 2$, we compute:

$$
\mathbf{A} - \lambda_{2/3} \mathbf{I} = \begin{bmatrix}
	2 - 2 & 0 & 0 \\
	0 & 3 - 2 & 1 \\
	0 & 1 & 3 - 2
\end{bmatrix} = \begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 1 \\
0 & 1 & 1
\end{bmatrix}
$$



#### Step 3: Solve $(\mathbf{A} - \lambda \mathbf{I}) \vec{v} = 0$ for each eigenvector
This step requires us to actually do some algebra and solve for the components of the eigenvectors.

<div class="two-column">

<div class="column">

The equation to solve is:<br>

$$
\begin{bmatrix}
	-2 & 0 & 0 \\
	0 & -1 & 1 \\
	0 & 1 & -1
\end{bmatrix}
\begin{bmatrix}
	a \\
	b \\
	c
\end{bmatrix} = \begin{bmatrix}
	0 \\
	0 \\
	0
\end{bmatrix}
$$

Expanding, we get:<br>

$$
\text{Row 1: } -2 a = 0 , \qquad \text{Row 2: } -b + c = 0
$$

and we can see that Row 3 will be redundant. <br>

Row 1 gives us $a = 0$, and Row 2 gives $b = c$.<br>

Thus, the general solution is:<br>

$$
\vec{v}_1 = \begin{bmatrix}
	0 \\
	b \\
	b
\end{bmatrix}
$$

where $b$ will be determined in the next step.

</div>

<div class="column">

The equation to solve is:<br>

$$
\begin{bmatrix}
	0 & 0 & 0 \\
	0 & 1 & 1 \\
	0 & 1 & 1
\end{bmatrix}
\begin{bmatrix}
	d \\
	e \\
	f
\end{bmatrix} = \begin{bmatrix}
	0 \\
	0 \\
	0
\end{bmatrix}
$$

Expanding, we get:<br>

$$
\text{Row 1: } 0 = 0 , \qquad \text{Row 2: } e + f = 0
$$

and Row 3 will be redundant. <br>

From Row 1, we can see that $d$ can be anything it wants, and Row 2 gives us $f = -e$.<br>

Thus, the general solution is:<br>

$$
\vec{v}_{2/3} = \begin{bmatrix}
	d \\
	e \\
	-e
\end{bmatrix}
$$

where $d$ and $e$ will be determined in the next step.

</div>

</div>







#### Step 4: Select the scale parameter for the eigenvector.



To simplify, we can take $b = 1$, giving:

$$
\vec{v}_1 = \begin{bmatrix}
	0 \\
	1 \\
	1
\end{bmatrix}
$$

This is the eigenvector associated with $\lambda_1 = 4$.


##### Now for the degenerate eigenvalue. 

When you have degenerate (i.e., repeated) eigenvalues, you have to be slightly more careful about what the undetermined constants you pick to fully determine the eigenvectors. If we have any choice about it, we would like it if the **eigenvectors to be mutually orthogonal** (perpendicular). So, what do we do?

First, pick coefficients to get one of our eigenvectors. For example, we could let $d = 1$ and $e = 1$ to get:

$$
\vec{v}_{2} = \begin{bmatrix}
	1 \\
	1 \\
	-1
\end{bmatrix}
$$

Great, we have one of the eigenvectors! But, how do we find the other one? 

By forcing the eigenvectors to be perpendicular! This means we force:

$$
\begin{aligned}
\vec{v}_3 \cdot \vec{v}_2 &= 0 \\[1.25ex]
\vec{v}_3^\text{T} \vec{v}_2 &= 0 \\[1.25ex]
\begin{bmatrix}
	d & e & -e
\end{bmatrix} \begin{bmatrix}
1 \\ 1 \\ -1
\end{bmatrix} &= 0 \\[4ex]
d + e + e &= 0 \\[1.25ex]
d &= -2e 
\end{aligned}
$$

Now we can let $e=1$, selected in a totally arbitrary manner, to get:

$$
\vec{v}_{3} = \begin{bmatrix}
	-2 \\
	1 \\
	-1
\end{bmatrix}
$$

So, we have the following eigenvalue/eigenvector pairs:

$$ \lambda_1 = 4 \,\,\,\, \vec{v}_1 = \begin{bmatrix}
	0 \\
	1 \\
	1
\end{bmatrix} \qquad \lambda_2 = 2 \,\,\,\, \vec{v}_2 = \begin{bmatrix}
1 \\
1 \\
-1
\end{bmatrix}  \qquad \lambda_3 = 2 \,\,\,\, \vec{v}_2 = \begin{bmatrix}
-2 \\
1 \\
-1
\end{bmatrix}  
$$

{% endcapture %}
{% include example.html content=ex %}








## Overcoming the Challenges of Lengthy Problems

As we can see, this process is quite involved, but none of the steps themselves are particularly difficult. It is human nature to equate the length of a project/activity, such as a math equation/problem or a term paper, to its perceived difficulty. However, this equating is often wrong. 

For instance, while the process we followed to find the eigenvectors had many steps, each individual step involved relatively straightforward mathematics. Go back and check for yourself. You will find nothing more than matrix multiplication and algebra. Often times people find "confusion" and "difficulty" of problems like this are in the number of steps it requires, which can feel overwhelming, but that does not make the problem itself difficult.

To help combat the feeling of being overwhelmed when working through lengthy math problems, here are a few tips:

- **Keep the goal in mind.**  
	Remember what you are trying to find/do. In this case, our goal was to determine the eigenvalues and eigenvectors of a matrix. Constantly reminding yourself of the end goal can make each step feel more purposeful and less daunting.
	
- **Break the problem into manageable pieces.**  
	Tackle each part of the problem one step at a time. For example, first write the characteristic equation, then solve for eigenvalues, then... Before you know it you will ave found all of the eigenvectors. Focusing on one small part at a time can make the problem feel more achievable.
	
- **Length doesn’t mean difficulty.**  
	A lengthy problem may involve multiple straightforward steps rather than inherently complex mathematics. Remind yourself that writing more doesn’t mean the math is more difficult, it generally just means you’re being thorough and are showing your work.
	
- **Stay organized.**  
	Clearly label each step of your solution and use consistent notation throughout. Keeping your work neat helps you avoid mistakes and makes it easier to review or spot errors.
	
- **Work systematically.**  
	Follow a logical sequence and resist the urge to jump around. Each step builds on the previous one, so working systematically ensures nothing is missed.
	
- **Take a mental break if needed.**  
	If you feel overwhelmed, pause for a moment, take a deep breath, and return to the problem with a fresh perspective. Lengthy problems are easier to manage with a calm and focused mindset.
	
- **Practice builds confidence.**  
	Like any skill, working through multi-step problems becomes easier with practice. The more you encounter lengthy problems, the more comfortable you will become in handling them.


By keeping these strategies in mind, you can approach even the most involved problems with confidence and clarity. Length and complexity are not the same, and with practice, what once felt overwhelming will become routine.






































## Application:

Now that we’ve learned how to solce the eigenvalue problem by finding the eigenvalues and eigenvectors, it’s time to explore their importance in the physical world and beyond. Eigenvalues and eigenvectors appear in a wide range of applications, from solving differential equations to describing the behavior of physical systems. Here, we’ll explore a few key applications to highlight their relevance.

Note: Some of these examples will resonate with you depending on where you are at in you physics journey. If you read an example and have no idea what it is talking about, don't panic! It just means there is a course in your future that will teach you something new and interesting. 

### Stability of Dynamical Systems

In physics and engineering, eigenvalues are often used to study the stability of dynamical systems. Consider a system of linear ordinary differential equations:

$$
\frac{d\vec{x}}{dt} = \mathbf{A} \vec{x}
$$

where $\vec{x}$ represents the state of the system, and $\mathbf{A}$ is the system's coefficient matrix. The eigenvalues of $\mathbf{A}$ determine the system’s stability:

- If all eigenvalues have negative real parts, the system is stable, as all solutions decay to equilibrium over time.  
- If any eigenvalue has a positive real part, the system is unstable, as solutions grow unbounded.  
- If eigenvalues have purely imaginary values, the system exhibits oscillatory behavior.  


For example, in coupled harmonic oscillators or predator-prey models, eigenvalues provide critical insight into whether the system will settle into a steady state, oscillate, or diverge.

We will work with this example specifically in Lecture 19. Stay tuned!

### Rotational Mechanics: Principal Axes of Rotation

Eigenvalues and eigenvectors are central to understanding the principal axes of physical systems. Consider the moment of inertia tensor in classical mechanics:

$$
\mathbf{I} = \begin{bmatrix}
	I_{xx} & I_{xy} & I_{xz} \\
	I_{yx} & I_{yy} & I_{yz} \\
	I_{zx} & I_{zy} & I_{zz}
\end{bmatrix}
$$

The eigenvectors of $\mathbf{I}$ represent the principal axes of rotation, and the eigenvalues correspond to the moments of inertia along these axes. By diagonalizing $\mathbf{I}$, we simplify rotational problems, reducing them to independent rotational motions about the principal axes.

### Quantum Mechanics: Observables and States

In quantum mechanics, eigenvalues and eigenvectors form the backbone of the whole theory. Observables, such as energy, position, and angular momentum, are represented by Hermitian operators (which can be represented as matrices), and the eigenvalues of these operators correspond to measurable quantities. For example:

- The Hamiltonian operator $ \hat{H} $ has eigenvalues representing the energy levels of a quantum system.  
- Eigenvectors of $ \hat{H} $, called eigenstates, describe the system's possible states, with their associated eigenenergy, when measured.  

This connection is fundamental to understanding quantum phenomena like atomic spectra and the behavior of spin systems.

### Vibration Analysis: Normal Modes

In systems with coupled oscillators, such as molecules or mechanical structures, eigenvalues and eigenvectors are used to determine the system's normal modes. For a system governed by:

$$
\mathbf{M} \frac{d^2\vec{x}}{dt^2} + \mathbf{K} \vec{x} = 0
$$

where $\mathbf{M}$ is the mass matrix and $\mathbf{K}$ is the stiffness matrix. The eigenvalues to this problem, as we have seen in Lecture 08, provide the squared frequencies of vibration, and the eigenvectors describe how the particles move for each mode.

### Data Science: Principal Component Analysis (PCA)

Eigenvalues and eigenvectors also play a crucial role in data analysis through techniques like Principal Component Analysis (PCA). PCA is used to reduce the dimensionality of large datasets by identifying the directions (eigenvectors) of maximum variance in the data. The eigenvalues determine the amount of variance captured along each principal component. This technique is widely used in machine learning, image processing, and genetics.
















## Problem:


- Please keep your work organized and neat.  
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.  
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.  


Consider the matrix:

$$
\mathbf{A} = 
\begin{bmatrix}
	2 & -1 \\
	-1 & 2
\end{bmatrix}
$$


a) Write out $(\mathbf{A} - \lambda \mathbf{I})$ as a single matrix.
	
b) Solve the characteristic equation $ \det(\mathbf{A} - \lambda \mathbf{I}) = 0 $ and find the eigenvalues $ \lambda $.
	
c) For each eigenvalue, find the corresponding eigenvector $ \vec{v} $.


















