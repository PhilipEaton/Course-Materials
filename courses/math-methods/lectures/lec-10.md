---
layout: default
title: Mathematical Methods - Lecture 10
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 10
---


# Lecture 10 – Introduction to Transforming into the Eigenbasis (Diagonalization)

In our previous discussions, we explored the eigenvalue problem, focusing on finding the eigenvalues and eigenvectors of matrices. These unique scalars and vectors provide critical insights into how a matrix acts on a vector space, revealing scaling factors and invariant directions. In this lecture, we take the concept further by investigating how eigenvectors can simplify matrix operations, particularly through transforming a matrix into its **eigenbasis** through a process called **digonalization**.

An interpretation of eigenvectors that we haven’t emphasized yet is that they represent the **optimal basis** for their associated matrix. To understand what we mean, let’s recall an assumption we made way back in Lecture 01. We assumed we could write $\hat{i}$, $\hat{j}$, and $\hat{k}$ in matrix form as:  

$$
\hat{i} = \begin{bmatrix}
	1 \\ 0 \\ 0
\end{bmatrix} \quad \quad 
\hat{j} = \begin{bmatrix}
	0 \\ 1 \\ 0
\end{bmatrix} \quad \quad 
\hat{k} = \begin{bmatrix}
	0 \\ 0 \\ 1
\end{bmatrix}
$$

We adopted these unit vectors as our standard basis because they offered a simple and intuitive way to represent vectors. However, this choice isn’t the only possible representation. Just as you can draw a coordinate system in different orientations on paper, you can choose different sets of basis vectors to describe the same vector space.

Eigenvectors represent the **optimal coordinate system representation** for the matrix they are associated with. Transforming into this eigenvector basis simplifies the matrix by restructuring it in a more transparent manner. Specifically, it allows us to **diagonalize the matrix**, where the diagonal entries become the eigenvalues. This diagonalized form is much easier to work with in calculations and provides direct insights into the matrix’s action.

We already know that transformations between coordinate systems can be expressed as matrices. In the case of eigenvectors, the transformation into the eigenvector basis is called an **eigenbasis transformation**. This transformation is an incredibly powerful tool in linear algebra and physics, enabling us to simplify systems, solve differential equations, and analyze complex interactions.









### Why Transform into the Eigenbasis?

A matrix’s eigenvectors define a special coordinate system where the action of the matrix becomes particularly simple. When expressed in its **eigenbasis**, a matrix will become a **diagonal matrix**, meaning it has nonzero entries only along its main diagonal:

$$
\mathbf{D} = 
\begin{bmatrix}
	\lambda_1 & 0 & \cdots & 0 \\
	0 & \lambda_2 & \cdots & 0 \\
	\vdots & \vdots & \ddots & \vdots \\
	0 & 0 & \cdots & \lambda_n
\end{bmatrix}
$$

where the diagonal entries $ \lambda_1, \lambda_2, \ldots, \lambda_n $ are the eigenvalues of the original matrix. Working with diagonal matrices is far easier than dealing with their original, often more complex, forms. For example:

- **Simplified matrix powers:** Powers of a diagonal matrix can be computed directly by raising each diagonal entry to the corresponding power. We have seen this in a previous lecture and will see an interesting application of this later in this lecture.
- **Simpler physical analysis:** Many physical systems become much easier to analyze when their governing matrices are diagonal. In this case, each eigenvalue often corresponds to a distinct mode or behavior of the system.
- **Quantum mechanics applications:** In quantum mechanics, diagonal matrices often correspond to measurements, where the eigenvalues represent the possible outcomes of the measurement.

This leads to a general rule of thumb for physicists and engineers: **If your working matrices are not diagonal, transform into the eigenbasis so that they are.** This simplification often reveals deeper insights into the underlying system.


{% capture ex %}

**A Note on Non-Commuting Matrices:** There are cases in physics and engineering where you will be working with multiple matrices at the same time. As a result, it may not be possible to diagonalize them using the same basis vectors. While this can be inconvenient, it is not insurmountable and often provides valuable information about the system. 

For instance, in quantum mechanics, the position and momentum operators cannot be simultaneously diagonalized because they do not commute. This is a direct consequence of the Heisenberg uncertainty principle. This incompatibility reflects fundamental properties of quantum systems and highlights why certain pairs of observables cannot be measured precisely at the same time.

In fact, this is a rule: Non-commuting matrices cannot be diagonalized at the same time using the same basis representation.

{% endcapture %}
{% include warning.html content=ex %}






### The Big Idea: Changing Basis

The key to diagonalizing a matrix is changing the basis in which it’s expressed. That is, we perform a coordinate transformation that moves the matrix from its current coordinate system into one defined by its eigenvectors.

Let’s pause briefly to explain what we mean by a “coordinate system representation” of a matrix. If we think of a matrix as an operator—like a rotation—that transforms coordinates, then the form that matrix takes depends on the coordinate system we're using. The operation itself doesn’t change, but its *appearance* does.

As a concrete example, consider the familiar 2D rotation matrix:

$$
\mathbf{R}(\theta) = \begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

Recall in Lecture 01 we agreed that the first element in the vector represented the $x$-component and similarly the second was the $y$-component:

$$
\vec{v} = \begin{bmatrix}
x \\ y
\end{bmatrix}
$$


When the rotation matrix is applied to this vector we get:

$$
\mathbf{R}(\theta) \vec{v} = \begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}\begin{bmatrix}
x \\ y
\end{bmatrix} = \begin{bmatrix}
x \cos(\theta) - y \sin(\theta) \\ x\sin(\theta) + y \cos(\theta)
\end{bmatrix}
$$

Now, let's suppose we change our mind. Suppose we want a new representation where the first element of a vector is the $y$-component and the second is the $x$-component:

$$
\vec{v}' = \begin{bmatrix}
y \\ x
\end{bmatrix}
$$

To preserve the meaning of the rotation, the matrix must change form. In this new representation, the rotation matrix becomes:

$$
\mathbf{R}'(\theta) = \begin{bmatrix}
\cos(\theta) & \sin(\theta) \\
-\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

such that:

$$
\mathbf{R}'(\theta) \vec{v}' = \begin{bmatrix}
\cos(\theta) & \sin(\theta) \\
-\sin(\theta) & \cos(\theta)
\end{bmatrix}\begin{bmatrix}
y \\ x
\end{bmatrix} = \begin{bmatrix}
y \cos(\theta) + x \sin(\theta) \\ - y \sin(\theta) + x \cos(\theta)
\end{bmatrix}
$$

Notice how the $x$ and $y$ components rotate just as they did in the original coordinate system. Only the *representation* of the operator has changed, not its effect.

The moral of the story is this: **the same matrix operation can look different depending on the coordinate system we use**. So, by choosing a new coordinate system we can transform a matrix into a much simpler form. If we transform into a coordinate system defined by the eigenvectors, that new form turns out to be diagonal. Let's see how this plays out in a general manner

Suppose $\mathbf{A}$ is a square $n \times n$ matrix with $n$ linearly independent eigenvectors $\vec{v}_1, \vec{v}_2, \ldots, \vec{v}_n$, and associated eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$. By definition, each eigenvector-eigenvalue pair satisfies the equation:

$$
\mathbf{A} \vec{v}_i = \lambda_i \vec{v}_i
$$

Let’s now construct a matrix $\mathbf{P}$ whose columns are these eigenvectors:

$$
\mathbf{P} = \begin{bmatrix}
\vert & \vert &  & \vert \\
\vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_n \\
\vert & \vert &  & \vert
\end{bmatrix}
$$

So, each column of $\mathbf{P}$ is one eigenvector of $\mathbf{A}$. Let’s look at what happens when we multiply $\mathbf{A}$ by $\mathbf{P}$:

$$
\mathbf{A} \mathbf{P} = \mathbf{A} \begin{bmatrix}
\vert & \vert &  & \vert \\
\vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_n \\
\vert & \vert &  & \vert
\end{bmatrix}
= \begin{bmatrix}
\vert & \vert &  & \vert \\
\mathbf{A} \vec{v}_1 & \mathbf{A} \vec{v}_2 & \cdots & \mathbf{A} \vec{v}_n \\
\vert & \vert &  & \vert
\end{bmatrix}
$$

---

To ``prove'' this we will look at a quick $2\times 2$ example. Consider the following matrix multiplication:

$$
\begin{bmatrix}
	1 & 2 \\ 3 & 4 
\end{bmatrix} \begin{bmatrix}
	2 & 3 \\ 4 & 5 
\end{bmatrix} = \begin{bmatrix}
10 & 13 \\ 22 & 29 
\end{bmatrix}
$$

Compare this what we get if we write the right hand matrix as two column vectors:

$$ \begin{bmatrix}
	1 & 2 \\ 3 & 4 
\end{bmatrix} \begin{bmatrix}
	2 & 3 \\ 4 & 5 
\end{bmatrix} =  \begin{bmatrix}
	1 & 2 \\ 3 & 4 
\end{bmatrix} \begin{bmatrix}
	\begin{bmatrix}
		2 \\ 4
	\end{bmatrix} & \begin{bmatrix}
		3 \\ 5 
	\end{bmatrix}
\end{bmatrix} = \begin{bmatrix} \begin{bmatrix}
	1 & 2 \\ 3 & 4 
\end{bmatrix} 
\begin{bmatrix}
	2 \\ 4
\end{bmatrix} & \begin{bmatrix}
1 & 2 \\ 3 & 4 
\end{bmatrix} \begin{bmatrix}
	3 \\ 5 
\end{bmatrix}
\end{bmatrix} = \begin{bmatrix}
\begin{bmatrix}
	10 \\ 22
\end{bmatrix} & \begin{bmatrix}
	13 \\ 29 
\end{bmatrix}
\end{bmatrix} $$


$$\implies  \begin{bmatrix}
	1 & 2 \\ 3 & 4 
\end{bmatrix} \begin{bmatrix}
	2 & 3 \\ 4 & 5 
\end{bmatrix} = \begin{bmatrix}
10 & 13 \\ 22 & 29 
\end{bmatrix} $$

We end up with the same result either way, which reinforces that multiplying a matrix by another matrix is like applying it to each column vector one at a time.

---

Now back to our main idea. Since $\mathbf{A} \vec{v}_i = \lambda_i \vec{v}_i$, we can rewrite the previous multiplication as:

$$
\mathbf{A} \mathbf{P} = \begin{bmatrix}
\vert & \vert &  & \vert \\
\lambda_1 \vec{v}_1 & \lambda_2 \vec{v}_2 & \cdots & \lambda_n \vec{v}_n \\
\vert & \vert &  & \vert
\end{bmatrix}
$$

And this can be factored as  (You are welcome to check by doing the matrix multiplication!):

$$
\begin{aligned}
\mathbf{A} \mathbf{P} &= \begin{bmatrix}
\vert & \vert &  & \vert \\
\vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_n \\
\vert & \vert &  & \vert
\end{bmatrix}
\begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n
\end{bmatrix}
= \mathbf{P} \mathbf{D}
\end{aligned}
$$

Here, $\mathbf{D}$ is a diagonal matrix containing the eigenvalues:

$$
\mathbf{D} = \begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n
\end{bmatrix}
$$

You might be wondering why we’d want to write things this way. Two good reasons:

1. It makes the next step of the derivation easier.
2. It highlights where the diagonal matrix $\mathbf{D}$ actually comes from.

Would you have thought to introduce $\mathbf{D}$ the first time you ever saw this problem? Probably not! It’s one of those tricks you pick up after working through the full process and realizing there’s a cleaner way to approach the problem. This is often the case when you see derivations or solutions that look "too clean" or "too clever". You might wonder how you were ever suppose to see these tricks on your own. The answer often is you weren't and the tricks were the result of someone spending months or years working on the problem until they stumbled on a "random" simplification that makes the process seem trivial. 

Anyways, let’s finish the transformation.

We’ve reached the following point:

$$
\mathbf{A} \mathbf{P} = \mathbf{P} \mathbf{D}
$$

We can solve for the diagonal matrix $\mathbf{D}$ by multiplying both sides on the left by $\mathbf{P}^{-1}$:

$$
\mathbf{P}^{-1} \mathbf{A} \mathbf{P} = \mathbf{D}
$$

This equation shows us how we can use $\mathbf{P}$ to transform $\mathbf{A}$ into its diagonal representation $\mathbf{D}$ by switching to the eigenvector basis.








### Conditions for Diagonalizability

Something we need to keep in mind for this lecture it that, similar to inevitability, **not all matrices can be diagonalized**. Even if a matrix has a non-zero determinant and can be inverted, that does not mean it can be diagonalized! 

Notice, in the above derivation we have to assume $\mathbf{P}$ had an inverse so that we could solve for the diagonal matrix $\mathbf{D}$. It turns out this is the condition required for a matrix to be diagonalized, 

A matrix $\mathbf{A}$ is diagonalizable **if and only if** it 

- has enough linearly independent eigenvectors to form a basis of the vector space. 
	- That is, the determinant of $\mathbf{A}$ is nonzero.
- the matrix $\mathbf{P}$ build from the eigenvectors is invertible.
	- That is, the determinant of $\mathbf{P}$ is nonzero.

It turns out for $\mathbf{P}$ to even be constructed, $\mathbf{A}$ must have a non zero determinate. So, assuming the matrix is invertible, we can test if it is diagonalizable by checking to see if the determinant of the matrix $\mathbf{P}$ is zero or not.

{% capture ex %}

**When is a matrix diagonalizable?** Suppose you have an $n\times n$ matrix $\mathbf{A}$ with eigenvalues $\lambda_1$, ..., $\lambda_n$, and eigenvectors $\vec{v}_1$, ..., $\vec{v}_n$. You can create the matrix $\mathbf{P}$ using the **eigenvectors** as its columns:

$$ \mathbf{P} = \begin{bmatrix}
	\vert & \vert & \cdots & \vert \\
	\vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_n\\
	\vert & \vert & \cdots & \vert
\end{bmatrix} 
$$

A matrix $\mathbf{A}$ will be diagonalizable if $\det(\mathbf{P}) \ne 0$, meaning all of the eigenvectors of matrix $\mathbf{A}$ are linearly independent. 
{% endcapture %}
{% include result.html content=ex %}

Some nice rules to keep in mind about the possible diagonalizability of a matrix has to do the its eigenvalues:

- If $\mathbf{A}$ has $n$ **distinct**, non-degenerate (i.e., non-repeated) eigenvalues, it is guaranteed to be diagonalizable. 
- If $\mathbf{A}$ has repeated eigenvalues, diagonalizability depends on the number of linearly independent eigenvectors associated with each eigenvalue. 
	- This is one of the major reasons we like the eigenvectors of repeated eigenvalues to be perpendicular. 

If $\mathbf{A}$ is not diagonalizable, we cannot express it in the form $\mathbf{A} = \mathbf{P} \mathbf{D} \mathbf{P}^{-1}$, where $\mathbf{D}$ is diagonal. Instead, we use a **Jordan normal form**, which involves generalized eigenvectors and blocks rather than a purely diagonal representation. This is beyond the scope of our discussion but is worth noting as a more general tool.











## Process of Transforming into the Eigenbasis


The process of transforming into the eigenbasis is ubiquitous in applications ranging from classical mechanics to quantum systems and from data analysis to control theory. Whether you're solving for normal modes of oscillation, simplifying a system of differential equations, or analyzing large datasets, diagonalization via the eigenbasis is often the go-to method to simplify the problem at hand. 

We will start with the mechanics of actually doing the transformation, and then we will explore its implications.




### Diagonalization Process

Transforming a matrix into its eigenbasis requires two primary steps: 
1) constructing the eigenvector matrix $ \mathbf{P} $, 
	- which means finding the eigenvalues and eigenvectors of the working matrix, and 
2) diagonalizing the original matrix $ \mathbf{A} $ using $\mathbf{D} = \mathbf{P}^{-1} \mathbf{A} \mathbf{P}$. 

In this section, we will carefully walk through this process and provide steps for carrying it out.



#### Step 1: Construct the Eigenvector Matrix $ \mathbf{P} $

The first step is to construct the matrix $ \mathbf{P} $, whose columns are the eigenvectors of $ \mathbf{A} $. That means we must first find the eigenvalues of the matrix, and then the eigenvectors. This is exactly what we have been working on in the last couple of lectures. 

As a reminder, we can find the eigenvalues ($ \lambda_1, \lambda_2, \ldots, \lambda_n $) by solving the characteristic equation:

$$
\det\left( \mathbf{A} - \lambda \mathbf{I} \right) = 0
$$

Then for each eigenvalue $ \lambda_i $, we can find the corresponding eigenvector $ \vec{v}_i $ by solving the system of equations:

$$
\left( \mathbf{A} - \lambda_i \mathbf{I} \right) \vec{v}_i = 0
$$

Once we have found all of the eigenvectors we can construct the transformation matrix $ \mathbf{P} $:

$$
\mathbf{P} = \begin{bmatrix} | & | &  & | \\ \vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_n \\ | & | &  & | \end{bmatrix}
$$


#### Step 1.5: Check is Diagonalization is Possible

Before we can just into using the diagonalization equation, we need to make sure it is even possible. This, as you will recall, is done by checking the determinant of the newly created $\mathbf{P}$ matrix. 

- If the determinant is zero, then diagonilization is NOT possible and we have to stop here. 
- If the determinant is nonzero, then diagonilization is possible and we can continue to Step 02. 



#### Step 2: Diagonalize the Matrix $ \mathbf{A} $

Once $ \mathbf{P} $ is constructed, and we have check its determinant, the next step is to use it to diagonalize $ \mathbf{A} $ using the diagonalization transformation:

$$
\mathbf{D} = \mathbf{P}^{-1} \mathbf{A} \mathbf{P}
$$

which is just a couple of matrix multiplications. You know you have it correct if you end up with:

$$
\mathbf{D} = 
\begin{bmatrix}
	\lambda_1 & 0 & \cdots & 0 \\
	0 & \lambda_2 & \cdots & 0 \\
	\vdots & \vdots & \ddots & \vdots \\
	0 & 0 & \cdots & \lambda_n
\end{bmatrix}
$$






## Examples of Transforming Matrices into the Eigenbasis

To better understand the mechanics of transforming matrices into their eigenbasis, let’s work through a detailed example. This process will reinforce the steps introduced earlier and highlight important nuances.

{% capture ex %}
Consider the symmetric matrix:

$$
\mathbf{A} = \begin{bmatrix}
4 & 2 \\
1 & 3
\end{bmatrix}
$$

**Step 1: Construct $\mathbf{P}$**

We begin by solving the characteristic equation to get the eigenvalues:

$$
\det\left( \mathbf{A} - \lambda \mathbf{I} \right) = 0
$$

Substituting $\mathbf{A}$ and $\mathbf{I}$, we have:

$$
\begin{aligned}
\text{det}(\mathbf{A} - \lambda \mathbf{I}) &= \begin{vmatrix}
4 - \lambda & 2 \\
1 & 3 - \lambda
\end{vmatrix}\\
&= (4 - \lambda)(3 - \lambda) - 2 \\
&= \lambda^2 - 7\lambda + 10 \\
&= (\lambda - 5 ) (\lambda - 2)\\
\end{aligned}
$$

Which gives us the characteristic equation:

$$
(\lambda - 5 ) (\lambda - 2) = 0
$$

and we find the eigenvalues:

$$
\lambda_1 = 5 \qquad \text{and} \qquad \lambda_2 = 2
$$

With the eigenvalues in hand, we can now solve for each eigenvector. 

<div class="two-column">

<div class="column">

For $\lambda_1 = 5$:<br>

$$
\begin{aligned}
(\mathbf{A} - \lambda_1 \mathbf{I})\vec{v} &= \vec{0} \\[1.25ex]
\begin{bmatrix}
	4 - (5) & 2 \\
	1 & 3 - (5)
\end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}\\[1.25ex]
\begin{bmatrix}
	-1 & 2 \\
	1 & -2
\end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\end{aligned}
$$

Row 1 gives:<br>

$$
-a + 2 b = 0 \implies a  = 2 b
$$

Choosing $b = 1$ leaves us with:<br>

$$
\vec{v}_1 = \begin{bmatrix}
2 \\ 1
\end{bmatrix}
$$

</div>

<div class="column">

For $\lambda_2 = 2$:<br>

$$
\begin{aligned}
(\mathbf{A} - \lambda_2 \mathbf{I})\vec{v} &= \vec{0} \\[1.25ex]
\begin{bmatrix}
	4 - (2) & 2 \\
	1 & 3 - (2)
\end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}\\[1.25ex]
\begin{bmatrix}
	2 & 2 \\
	1 & 1
\end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\end{aligned}
$$

Row 1 gives:<br>

$$
2a + 2b = 0 \implies b = - a
$$

Choosing $a = 1$ leaves us with:<br>

$$
\vec{v}_2 = \begin{bmatrix}
1 \\ -1
\end{bmatrix}
$$

</div>

</div>


Finally armed with the eigenvectors, we can construct the eigenvector matrix:

$$
\mathbf{P} = \begin{bmatrix} | & |  \\ \vec{v}_1 & \vec{v}_2  \\ | & |  \end{bmatrix} \qquad \implies \qquad 
\mathbf{P} =  \begin{bmatrix}
2 & 1 \\
1 & -1
\end{bmatrix}
$$

Now, let's check that the diagonalization is even possible. Taking the determinant of $\mathbf{P}$ gives:

$$ 
 \begin{vmatrix}
2 & 1 \\
1 & -1
\end{vmatrix} = -2 - 1 = -3
$$

This determinant is nonzero, meaning the digonalization is possible. 

Since we will need it, it would be useful to get $\mathbf{P}^{-1}$ now. Using a computer we get:

$$
\mathbf{P}^{-1} = - \frac{1}{3 }
\begin{bmatrix}
-1 & -1 \\
-1 & 2
\end{bmatrix}
$$


**Step 2: Apply the Diagonalization Formula**


The diagonal matrix is:

$$
\begin{aligned}
\mathbf{D} &= \mathbf{P}^{-1} \mathbf{A} \mathbf{P} \\[1.25ex]
&= -\frac{1}{3 }
\begin{bmatrix}
-1 & -1 \\
-1 & 2
\end{bmatrix} \begin{bmatrix}
4 & 2 \\
1 & 3
\end{bmatrix}  \begin{bmatrix}
2 & 1 \\
1 & -1
\end{bmatrix} \\[1.25ex]
&= -\frac{1}{3 }
\begin{bmatrix}
-1 & -1 \\
-1 & 2
\end{bmatrix} \begin{bmatrix}
10  & 2 \\
5 & -2
\end{bmatrix} \\[1.25ex]
&= - \frac{1}{3 }
\begin{bmatrix}
-15  & -2 + 2    \\
	-10 + 10 & -2 - 4    \\
\end{bmatrix}  \\[1.25ex]
&= 
\begin{bmatrix}
5  & 0    \\
0 & 2    \\
\end{bmatrix}  \\[1.25ex]
\end{aligned}
$$

Notice, the eigenvalues we found previously are the values along the main diagonal. This supports the conclusion that we have performed the diagonalization process correctly. 
{% endcapture %}
{% include example.html content=ex %}





{% capture ex %}
Consider the symmetric matrix:

$$
\mathbf{A} = 
\begin{bmatrix}
2 & 1 \\
1 & 4
\end{bmatrix}
$$

**Step 1: Construct $\mathbf{P}$**

We begin by solving the characteristic equation to get the eigenvalues:

$$
\det\left( \mathbf{A} - \lambda \mathbf{I} \right) = 0
$$

Substituting $\mathbf{A}$ and $\mathbf{I}$, we have:

$$
\begin{aligned}
\text{det}(\mathbf{A} - \lambda \mathbf{I}) &= 0\\
\begin{vmatrix}
2 - \lambda & 1 \\
1 & 4 - \lambda
\end{vmatrix} &= 0\\
 (2 - \lambda)(4 - \lambda) -1 &= 0\\
 \lambda^2 - 6\lambda + 7 &= 0
\end{aligned}
$$

Solving the quadratic equation, we find the eigenvalues:

$$
\lambda_1 = 3 + \sqrt{2} \qquad \text{and} \qquad \lambda_2 = 3 - \sqrt{2}
$$

With the eigenvalues in hand, we can now solve for each eigenvector. 

<div class="two-column">

<div class="column">

For $\lambda_1 = 3 + \sqrt{2}$:<br>

$$
\begin{aligned}
(\mathbf{A} - \lambda_1 \mathbf{I})\vec{v} &= \vec{0} \\[1.25ex]
\begin{bmatrix}
	2 - (3 + \sqrt{2}) & 1 \\
	1 & 4 - (3 + \sqrt{2})
\end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}\\[1.25ex]
\begin{bmatrix}
	-1 - \sqrt{2}  & 1 \\
	1 & 1 - \sqrt{2} 
\end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\end{aligned}
$$ 

Row 1 gives:<br>

$$
-(1 + \sqrt{2} ) a +  b = 0 \implies b  = (1+\sqrt{2}) a
$$

Choosing $a = 1$ leaves us with:<br>

$$
\vec{v}_1 = \begin{bmatrix}
1 \\ 1 + \sqrt{2}
\end{bmatrix}
$$

</div>

<div class="column">

For $\lambda_2 = 3 - \sqrt{2}$:<br>

$$
\begin{aligned}
(\mathbf{A} - \lambda_2 \mathbf{I})\vec{v} &= \vec{0} \\[1.25ex]
\begin{bmatrix}
	2 - (3 - \sqrt{2}) & 1 \\
	1 & 4 - (3 - \sqrt{2})
\end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}\\[1.25ex]
\begin{bmatrix}
	-1 + \sqrt{2} & 1 \\
	1 & 1 + \sqrt{2} 
\end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\end{aligned}
$$

Row 1 gives:<br>

$$
(-1 + \sqrt{2} )a + b = 0 \implies b = (1 - \sqrt{2}) a
$$

Choosing $a = 1$ leaves us with:<br>

$$
\vec{v}_2 = \begin{bmatrix}
1 \\ 1 - \sqrt{2} 
\end{bmatrix}
$$

</div>

</div>


Finally armed with the eigenvectors, we can construct the eigenvector matrix:

$$
\mathbf{P} = \begin{bmatrix} | & |  \\ \vec{v}_1 & \vec{v}_2  \\ | & |  \end{bmatrix} \qquad \implies \qquad 
\mathbf{P} =  \begin{bmatrix}
1 & 1 \\
1 + \sqrt{2} & 1 - \sqrt{2}
\end{bmatrix}
$$

Let's check to see if diagonalization is even possible by tkaing the determinant of $\mathbf{P}$:

$$ 
\begin{vmatrix}
1 & 1 \\
1 + \sqrt{2} & 1 - \sqrt{2}
\end{vmatrix} = (1 - \sqrt{2}) - (1 + \sqrt{2}) = -2 \sqrt{2}
$$

The determinant is nonzero, so diagonilization is possible.

Since we will have need for it in the next step, lets get $\mathbf{P}^{-1}$ now (using a computer/calculator):

$$
\mathbf{P}^{-1} = - \frac{1}{2\sqrt{2}}
\begin{bmatrix}
1 - \sqrt{2} & -1 \\
-1 - \sqrt{2} & 1
\end{bmatrix}
$$

**Step 2: Apply the Diagonalization Formula**

The diagonal matrix is:

$$
\begin{aligned}
\mathbf{D} &= \mathbf{P}^{-1} \mathbf{A} \mathbf{P} \\[1.25ex]
&= - \frac{1}{2\sqrt{2}}
\begin{bmatrix}
1 - \sqrt{2} & -1 \\
-1 - \sqrt{2} & 1
\end{bmatrix} \begin{bmatrix}
2 & 1 \\
1 & 4
\end{bmatrix} \begin{bmatrix}
1 & 1 \\
1 + \sqrt{2} & 1 - \sqrt{2}
\end{bmatrix} \\[1.25ex]
&= - \frac{1}{2\sqrt{2}}
\begin{bmatrix}
1 - \sqrt{2} & -1 \\
-1 - \sqrt{2} & 1
\end{bmatrix} \begin{bmatrix}
3 + \sqrt{2}   & 3 - \sqrt{2}  \\
5 + 4 \sqrt{2}  & 5 - 4 \sqrt{2}
\end{bmatrix} \\[1.25ex]
&= - \frac{1}{2\sqrt{2}}
\begin{bmatrix}
1 - 2 \sqrt{2} - 5 - 4 \sqrt{2}   & 5 -4 \sqrt{2} - 5 + 4\sqrt{4}     \\
-5 -4 \sqrt{2} + 5 + 4 \sqrt{2}  & -1 - 2 \sqrt{2} + 5 - 4 \sqrt{2}     \\
\end{bmatrix}  \\[1.25ex]
&= - \frac{1}{2\sqrt{2}}
\begin{bmatrix}
-4 - 6 \sqrt{2}   & 0    \\
0 & 4 - 6 \sqrt{2}     \\
\end{bmatrix}  \\[1.25ex]
&= - \frac{1}{2\sqrt{2}}
\begin{bmatrix}
2\sqrt{2}(-\sqrt{2} - 3)   & 0    \\
0 & 2\sqrt{2}(\sqrt{2} - 3)     \\
\end{bmatrix}  \\[1.25ex]
&= 
\begin{bmatrix}
3 + \sqrt{2}   & 0    \\
0 & 3 -\sqrt{2}    \\
\end{bmatrix}
\end{aligned}
$$

which are the eigenvalues we found for each of the eigenvectors. 
{% endcapture %}
{% include example.html content=ex %}











## Applications

Diagonalization is not just a mathematical curiosity, it has profound implications in both theoretical and applied contexts. By transforming a matrix into its eigenbasis, we can simplify computations, reveal intrinsic properties, and gain deeper insights into physical systems. Let’s explore some key applications.



### Simplifying Matrix Operations

When a matrix $\mathbf{A}$ is diagonalized, its diagonal form $\mathbf{D}$ allows for some straightforward computations. We know $\mathbf{A}$ and $\mathbf{D}$ are linked through the eigenvector matrix $\mathbf{P}$ in the following manner:

$$
\mathbf{A} = \mathbf{P} \mathbf{D} \mathbf{P}^{-1}
$$

In this form, Calculating the matrix $\mathbf{A}$ raised to a power becomes significantly easier. Consider the following:

$$ \mathbf{A}^2 =  (\mathbf{P} \mathbf{D} \mathbf{P}^{-1})^2 = \mathbf{P} \mathbf{D} \underbrace{\mathbf{P}^{-1} \,\,   \mathbf{P}}_{= \mathbf{I}} \mathbf{D} \mathbf{P}^{-1} = \mathbf{P} \mathbf{D}\mathbf{D} \mathbf{P}^{-1} = \mathbf{P} \mathbf{D}^2 \mathbf{P}^{-1}$$

which we can readily generalize to:

$$
\mathbf{A}^n = (\mathbf{P} \mathbf{D} \mathbf{P}^{-1})^n = \mathbf{P} \mathbf{D}^n \mathbf{P}^{-1}
$$

As another example, consider the exponential where the power is the matrix $\mathbf{A}$:

$$e^{\mathbf{A}} =?$$

Presently, we only know how to multiple matrices together (well that and how to add and subtract them). We know nothing about how to handle having a matrix in the exponent. 

So, if we can convert from the exponential form to some matrix multiplications, we would be in business. But, how do we do this? 

By doing the MacLaurin Expansion! In fact, any smooth function can be changed to a series of multiplications (i.e., polynomials) via expansions. 

Performing this expansion gives:

$$ e^{\mathbf{A}}  = 1 + \frac{1}{1!} \mathbf{A} + \frac{1}{2!} \mathbf{A}^2 + \frac{1}{3!} \mathbf{A}^3 + \cdots $$

We can use what we just learn to trade $\mathbf{A}^n$ with $\mathbf{P} \mathbf{D}^n \mathbf{P}^{-1}$ to get:

$$ e^{\mathbf{A}}  = 1 + \frac{1}{1!} \mathbf{P} \mathbf{D} \mathbf{P}^{-1} + \frac{1}{2!} \mathbf{P} \mathbf{D}^2 \mathbf{P}^{-1} + \frac{1}{3!} \mathbf{P} \mathbf{D}^3 \mathbf{P}^{-1} + \cdots $$

Notice, we can factor $\mathbf{P}$ out on the left and $\mathbf{P}^{-1}$ on the right:

$$ e^{\mathbf{A}}  = \mathbf{P} \left( 1 + \frac{1}{1!}  \mathbf{D}  + \frac{1}{2!}  \mathbf{D}^2  + \frac{1}{3!}  \mathbf{D}^3  + \cdots \right) \mathbf{P}^{-1}$$

which can be expressed as:

$$ e^{\mathbf{A}}  = \mathbf{P} e^{\mathbf{D}}  \mathbf{P}^{-1}$$

Interestingly, we can repeat this process for any smooth function to get:

$$ f(\mathbf{A})  = \mathbf{P} f(\mathbf{D})  \mathbf{P}^{-1}$$

These, and similar, simplifications are invaluable in physics, particularly in solving differential equations and analyzing time evolution in quantum mechanics.





### Decoupling Systems of Differential Equations

Many physical systems are described by coupled differential equations. Diagonalization allows us to decouple these systems into independent equations. 

For example, consider a set of linear differential equations:

$$
\frac{d\vec{x}}{dt} = \mathbf{A} \vec{x}
$$

By diagonalizing $\mathbf{A}$, we can transform the system into:

$$
\frac{d\vec{y}}{dt} = \mathbf{D} \vec{y}
$$

where $\vec{y} = \mathbf{P}^{-1} \vec{x}$. This reduces the problem to solving independent differential equations for each component of $\vec{y}$. We will see examples of this in Lecture 19.





### Analyzing Quantum Systems

In quantum mechanics, diagonalization is essential for solving the Schrödinger equation. The Hamiltonian matrix $\mathbf{H}$, which represents the total energy operator of a system, is diagonalized to find its eigenvalues and eigenstates:

$$
\mathbf{H} \psi_n = E_n \psi_n
$$

Here:

- The eigenvalues $E_n$ correspond to the energy levels of the quantum system.
- The eigenstates $\psi_n$ form a complete orthonormal basis for describing the system.


Diagonalization simplifies the analysis of quantum systems, enabling us to predict physical observables and time evolution.











## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


In quantum mechanics, the Hamiltonian operator $ \hat{H} $ determines the total energy of a system. Suppose we are working with a two-level quantum system, such as a spin-$\frac{1}{2}$ particle in an external magnetic field. The Hamiltonian is represented by the following $ 2 \times 2 $ matrix:

$$
\mathbf{H} = \begin{bmatrix}
	\Delta & \gamma \\
	\gamma & -\Delta
\end{bmatrix}
$$

where:

- $ \Delta $ is a real number representing the energy splitting of the two levels in the absence of interactions.
- $ \gamma $ is a real coupling constant describing the interaction between the levels.
- Row 1, Column 1 corresponds to state 1 of the quantum system with an energy splitting given by $\Delta$. 
- Row 2, Column 2 corresponds to state 2 of the quantum system with an energy splitting given by $-\Delta$. 
- Row 1, Column 2 represents the shift in energy of state 1 due to possible interactions coming from state 2.
- Row 2, Column 1 represents the shift in energy of state 2 due to possible interactions coming from state 1.


In the case $\gamma \ne 0$ we would say states 1 and 2 are coupled together. In this problem we will see if we can find a representation where our new working states are decoupled. 

For this problem, you can take $\Delta = 3$ and $\gamma = 4$. 



a) Find the eigenvalues of the Hamiltonian matrix $ \mathbf{H} $. What do you think these eigenvalues represent in the context of the quantum system?  

b) Determine the eigenvectors of $ \mathbf{H} $.   

c) Construct the matrix $ \mathbf{P} $, whose columns are the eigenvectors of $ \mathbf{H} $, and verify that $ \mathbf{P}^{-1} \mathbf{H} \mathbf{P} $ is diagonal. What are the diagonal entries of this matrix, and what do you think they represent?














