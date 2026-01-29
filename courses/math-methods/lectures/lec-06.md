---
layout: default
title: Mathematical Methods - Lecture 06
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 6
---

# Lecture 06 – Orthogonal Matrices Part 2 and Linear Dependence/Independence

Let’s continue our discussion of orthogonal matrices by exploring some of their key properties and why they matter so much in physics.

## Why do we care about orthogonal matrices?

Orthogonal matrices have a special feature:

> Orthogonal matrix operations preserve both the **magnitude** of a vector and the **angle** between two vectors.

If you're looking for a new math buzz word, this property is called an *isometry*.

This preservation is important in physics. Why? Because matrices often represent coordinate transformations. And in the real world, when we change our perspective, like turning our head or rotating in our swivel chairs, we don’t expect the actual distance between objects or the angles between position vectors to change. That wouldn't agree with our observations of the world in non-relativistic situations. When things get up to relativistic speeds, things change and oddities crop up.

{% capture ex %}
Let's do a sanity check of these claimed properties of an orthogonal matrix. We saw previously that rotation matrices are orthogonal. So, let's apply a rotation matrix to a pair of vectors and verify the magnitudes of the vectors and that the angle between them remain unchanged .

Let’s use the orthogonal matrix $ \mathbf{R}_z(\theta) $, which represents a rotation in three dimensions about the $ z $-axis by an angle $ \theta $:

$$
\mathbf{R}_z(\theta) = \begin{bmatrix}
	\cos(\theta) & -\sin(\theta) & 0 \\
	\sin(\theta) & \cos(\theta) & 0 \\
	0 & 0 & 1
\end{bmatrix}
$$

Now consider a vector $ \vec{v} $:

$$
\vec{v} = \begin{bmatrix}
	v_x \\
	v_y \\
	v_z
\end{bmatrix}
$$

When we apply the rotation matrix $ \mathbf{R}_z(\theta) $ to $ \vec{v} $, we get a new vector $ \vec{v}' $:

$$
\vec{v}' = \mathbf{R}_z(\theta) \vec{v} = \begin{bmatrix}
	v_x \cos(\theta) - v_y \sin(\theta) \\
	v_x \sin(\theta) + v_y \cos(\theta) \\
	v_z
\end{bmatrix}
$$

Let’s now look at the magnitudes of these vectors and see if they are the same. The magnitude of $ \vec{v} $ is:

$$
\lvert \vec{v} \rvert = \sqrt{v_x^2 + v_y^2 + v_z^2}
$$

and the magnitude of $ \vec{v}' $ simplifies as follows:

$$
\begin{aligned}
	\lvert \vec{v}'\rvert &= \sqrt{(v_x \cos(\theta) - v_y \sin(\theta))^2 + (v_x \sin(\theta) + v_y \cos(\theta))^2 + v_z^2} \\[1.0ex]
	&= \sqrt{\begin{aligned} &v_x^2 \cos^2(\theta) - 2 v_x v_y \cos(\theta)\sin(\theta) + v_y^2 \sin^2(\theta) + v_x^2 \sin^2(\theta)\\ 
							 &+ 2 v_x v_y \sin(\theta)\cos(\theta) + v_y^2 \cos^2(\theta) + v_z^2
			\end{aligned}} \\[1.15ex]
	&= \sqrt{\begin{aligned} &v_x^2 (\cos^2(\theta) + \sin^2(\theta)) - 2 v_x v_y (\cos(\theta)\sin(\theta) + \sin(\theta)\cos(\theta))  \\
							 &+ v_y^2 ( \sin^2(\theta) + \cos^2(\theta)) + v_z^2
			 \end{aligned}} \\[1.0ex]
	\lvert \vec{v}' \rvert &= \sqrt{v_x^2  + v_y^2 + v_z^2}
\end{aligned}
$$

So, the magnitude of the vector is unchanged after being rotated!

Next, let’s examine how the angle between two vectors is affected. Consider another vector $ \vec{u} $:

$$
\vec{u} = \begin{bmatrix}
	u_x \\
	u_y \\
	u_z
\end{bmatrix}
$$

Remember, the dot product depends on both magnitudes and the angle between two vectors, and we just showed that the rotation leaves magnitudes unchanged. If we can show that the dot product also remains unchaged, then that is confirmation that the angle between two vectors is the same after rotating of both vectors. 

The dot product of $ \vec{u} $ and $ \vec{v} $ is:

$$
\vec{u} \cdot \vec{v} = u_x v_x + u_y v_y + u_z v_z = \vec{u}^\text{T} \vec{v}
$$

Applying the rotation matrix to both vectors gives:

$$
\vec{u}' = \mathbf{R}_z(\theta) \vec{u} \quad \text{and} \quad \vec{v}' = \mathbf{R}_z(\theta) \vec{v}
$$

Then their dot product becomes:

$$
\vec{u}' \cdot \vec{v}' = (\mathbf{R}_z(\theta) \vec{u})^\text{T} (\mathbf{R}_z(\theta) \vec{v})
$$

Now we can use the transpose identity (which we will explain in detail soon):

$$
(\mathbf{A} \mathbf{B})^\text{T} = \mathbf{B}^\text{T} \mathbf{A}^\text{T} \quad \Rightarrow \quad (\mathbf{R}_z(\theta) \vec{u})^\text{T} = \vec{u}^\text{T} \mathbf{R}_z(\theta)^\text{T}
$$

Substituting this in:

$$
\vec{u}' \cdot \vec{v}' = \vec{u}^\text{T} \mathbf{R}_z(\theta)^\text{T} \mathbf{R}_z(\theta) \vec{v}
$$

Because $ \mathbf{R}_z(\theta) $ is orthogonal, we know:

$$
\mathbf{R}_z(\theta)^\text{T} \mathbf{R}_z(\theta) = \mathbf{I}
$$

So:

$$
\vec{u}' \cdot \vec{v}' = \vec{u}^\text{T} \mathbf{I} \vec{v} = \vec{u}^\text{T} \vec{v} = \vec{u} \cdot \vec{v}
$$

This result confirms that the angle between the vectors is also preserved. We have given general proofs of these properties below, for those who are curious.

{% endcapture %}
{% include example.html content=ex %}

### Properties of Orthoginal Matrices

To conclude, orthogonal matrices have several important mathematical and physical properties, which make them particularly useful in various applications. Below are some of their key properties, new and old:


- **Transpose Equals Inverse:**   An orthogonal matrix $\mathbf{A}$, by definition, satisfies:  

	$$
	\mathbf{A}^\text{T} \mathbf{A} = \mathbf{A} \mathbf{A}^\text{T} = \mathbf{I}
	$$

	where $\mathbf{I}$ is the identity matrix. This means the transpose of $\mathbf{A}$ is its inverse: $\mathbf{A}^{-1} = \mathbf{A}^\text{T}$.


- **Column and Row Orthogonality:** The columns (and rows) of an orthogonal matrix are orthonormal (orthogonal and of unit magnitude). Specifically:  

	$$\mathbf{A} = \begin{bmatrix}
		\vert & \vert & \cdots \\
		\vec{a}_1 & \vec{a}_2 & \cdots\\
		\vert & \vert & \cdots
	\end{bmatrix} \qquad \text{or}  \qquad \mathbf{A} = \begin{bmatrix}
		-  \vec{a}_1^\text{T} - \\ -  \vec{a}_2^\text{T} -  \\ \cdots
	\end{bmatrix} \qquad \qquad
	\vec{a}_i \cdot \vec{a}_j = \begin{cases}
		1 & \text{if }  i = j \\
		0 & \text{if }  i \ne j 
	\end{cases}
	$$

- **Norm Preservation:** Orthogonal matrices preserve the magnitude of vectors. For any vector $\vec{v}$:  

	$$
	\lvert \vec{v} \rvert = \lvert \mathbf{A} \vec{v}\rvert
	$$

	We can prove this in general in the fllowing manner:
	
	$$
	\lvert \mathbf{A} \vec{v}\rvert^2 =  (\mathbf{A} \vec{v}) \cdot (\mathbf{A} \vec{v}) = (\mathbf{A} \vec{v})^\text{T} (\mathbf{A} \vec{v}) =  \vec{u}^\text{T} \mathbf{A}^\text{T} \mathbf{A} \vec{v} = \vec{v}^\text{T} \mathbf{I}\vec{v} = \vec{v}^\text{T} \vec{v} = \vec{v} \cdot \vec{v} = \lvert \vec{v} \rvert^2
	$$

- **Angle Preservation:** Orthogonal matrices preserve the angles between vectors. If $\vec{u}$ and $\vec{v}$ are two vectors, then:  

	$$
	(\mathbf{A} \vec{u}) \cdot (\mathbf{A} \vec{v}) = \vec{u} \cdot \vec{v}
	$$

	This one is easy to prove:

	$$
	(\mathbf{A} \vec{u}) \cdot (\mathbf{A} \vec{v}) = (\mathbf{A} \vec{u})^\text{T} (\mathbf{A} \vec{v}) =  \vec{u}^\text{T} \mathbf{A}^\text{T} \mathbf{A} \vec{v} = \vec{u}^\text{T} \mathbf{I}\vec{v} = \vec{u}^\text{T} \vec{v} = \vec{u} \cdot \vec{v}
	$$

- **Determinant:** It turns out, for the previous two properties to be true, it must be that the determinant of an orthogonal matrix is either $+1$ (for proper rotations) or $-1$ (for improper rotations, which include reflections):  

	$$
	\det(\mathbf{A}) = \pm 1
	$$

	We will state this without proof, and reference the first few lectures where we discussed the determinant and its relation to rescaling (which will change angles between vectors) and reflections. 
















### Uses of Orthogonal Matrices

Orthogonal matrices are used in physics in various different manners. Here are some common, some new and some old, examples of where we would use these kinds of matrices:

- **Coordinate Transformations:** Orthogonal matrices are used to describe rotations and reflections in space.
- **Conservation of Volume:** Proper orthogonal matrices (those with $\det(\mathbf{A}) = +1$) describe transformations that preserve volume in $n$-dimensional space.
- **Symmetry Operations:** Orthogonal matrices can represent symmetry operations, such as those used in crystallography and molecular physics.
- **Eigenvalue Properties:** The eigenvalues of an orthogonal matrix lie on the unit circle in the complex plane, with magnitudes equal to 1. We will see this in few lectures. 
- **Time Reversal or Parity in Physics:**  Improper orthogonal matrices (with $\det(\mathbf{A}) = -1$) can describe transformations like parity (spatial inversion) and time reversal.










## Special Properties of Matrix Operations

Before moving on to Linear Dependence and Independence, which is a major part of linear algebra and will occupy the remainder of this lecture and all of the next, let's address a gap in matrix operations we saw, but did not prove earlier.

### Inverse Flipping

Consider the following matrix operation:  
$$
\mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D}  \vec{v} = \vec{v}'
$$
where $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, and $\mathbf{D}$ are all square matrices with inverses. One might ask the following: 


> How can we manipulate this equation to solve for $\vec{v}$? 


In other words, how do we perform the linear algebra necessary to isolate $\vec{v}$? This process provides a fun example of the "algebra" part of "linear algebra" coming into play.

To solve for $\vec{v}$, we need to isolate the vector by removing the matrices acting on it. However, we cannot simply apply the inverse of $\mathbf{D}$, i.e., $\mathbf{D}^{-1}$, to move it to the other side. This is because if we try to apply $\mathbf{D}^{-1}$ on the left, the other matrices interfere, and matrices do not necessarily commute. That is, we cannot swap the order of the matrices willy nilly. Similarly, if we apply $\mathbf{D}^{-1}$ on the right, $\vec{v}$ gets in the way.

Therefore, we must remove $\mathbf{A}$ first by applying its inverse on the left:

$$
\begin{aligned}
	\mathbf{A}^{-1} \mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D}  \vec{v} &= \mathbf{A}^{-1} \vec{v}' \\
	\mathbf{I} \mathbf{B} \mathbf{C} \mathbf{D}  \vec{v} &= \mathbf{A}^{-1} \vec{v}' \\
	\mathbf{B} \mathbf{C} \mathbf{D}  \vec{v} &= \mathbf{A}^{-1} \vec{v}'
\end{aligned}
$$

where $\mathbf{I}$ is the identity matrix. We can move the other matrices in a similar manner:

$$
\begin{aligned}
	\mathbf{B}^{-1}\mathbf{B} \mathbf{C} \mathbf{D}  \vec{v} &= \mathbf{B}^{-1}\mathbf{A}^{-1}\vec{v}'\\
	\mathbf{I} \mathbf{C} \mathbf{D}  \vec{v} &= \mathbf{B}^{-1}\mathbf{A}^{-1}\vec{v}'\\
	\mathbf{C} \mathbf{D}  \vec{v} &= \mathbf{B}^{-1}\mathbf{A}^{-1}\vec{v}'\\
	\mathbf{C}^{-1}\mathbf{C} \mathbf{D}  \vec{v} &= \mathbf{C}^{-1}\mathbf{B}^{-1}\mathbf{A}^{-1}\vec{v}'\\
	\mathbf{I} \mathbf{D}  \vec{v} &= \mathbf{C}^{-1} \mathbf{B}^{-1} \mathbf{A}^{-1}\vec{v}'\\
	\mathbf{D}  \vec{v} &= \mathbf{C}^{-1} \mathbf{B}^{-1} \mathbf{A}^{-1}\vec{v}'\\
	\mathbf{D}^{-1}\mathbf{D}  \vec{v} &= \mathbf{D}^{-1} \mathbf{C}^{-1} \mathbf{B}^{-1} \mathbf{A}^{-1}\vec{v}'\\
	\mathbf{I}  \vec{v} &= \mathbf{D}^{-1} \mathbf{C}^{-1} \mathbf{B}^{-1} \mathbf{A}^{-1}\vec{v}'\\
	\vec{v} &= \mathbf{D}^{-1} \mathbf{C}^{-1} \mathbf{B}^{-1} \mathbf{A}^{-1}\vec{v}'
\end{aligned}
$$

Like peeling the layers of an onion, the outer most operation must be removed first and we work our way inwards. If we did this all in one step it would look something like this:

$$
\begin{aligned}
	\mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D}  \vec{v} &= \vec{v}' \\
	(\mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D})^{-1} \mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D}  \vec{v} &= (\mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D})^{-1} \vec{v}' \\
	\mathbf{I}  \vec{v} &= (\mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D})^{-1}\vec{v}' \\
	\vec{v} &= (\mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D})^{-1} \vec{v}' 
\end{aligned}
$$

Comparing the two answer we can see that this means: 

$(\mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D})^{-1} = \mathbf{D}^{-1} \mathbf{C}^{-1} \mathbf{B}^{-1} \mathbf{A}^{-1}$. Notice how the order of the matrices flip as a result of taking the inverse, an important observation that is often not stressed enough.

{% capture ex %}

Taking the inverse of multiple matrices is identical to taking the reverse order of the inverse of each individual matrix:

$$ ( \mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D} )^{-1} = \mathbf{D}^{-1} \mathbf{C}^{-1} \mathbf{B}^{-1} \mathbf{A}^{-1} $$

{% endcapture %}
{% include result.html content=ex %}

Recall for orthogonal matrices, their inverse is also their transpose. Therefore, if applying the inverse flips the order of the matrices in this manner, the transpose must do the same, or else orthogonal matrices would not behave as expected!

{% capture ex %}

Taking the transpose of multiple matrices is identical to taking the reverse order of the transpose of each individual matrix:

$$ ( \mathbf{A} \mathbf{B} \mathbf{C} \mathbf{D} )^\text{T} = \mathbf{D}^\text{T} \mathbf{C}^\text{T} \mathbf{B}^\text{T} \mathbf{A}^\text{T} $$

{% endcapture %}
{% include result.html content=ex %}












## Special Orthogonal and Unitary Groups

Having explored orthogonal matrices and their properties, we can now introduce two particularly important groups used in physics: the **Special Orthogonal Group** ($SO$) and the **Special Unitary Group** ($SU$). These groups are orthogonal, as the name suggests, which means they preserve key physical quantities, such as lengths, angles, or probabilities (in quantum mechanics), while adding structure like orientation or phase invariance (another quantum thing).

All orthogonal matrices together form the *orthogonal group*, denoted $O(n)$, where $n$ indicates the number of dimensions. For example, $O(2)$ includes all $2 \times 2$ orthogonal matrices. Remember, to be orthogonal a matrix must satisfy:

$$ 
\mathbf{A}^{-1} = \mathbf{A}^\text{T} 
$$

A consequence of this is that every orthogonal matrix has a determinant of either $+1$ or $-1$, as we have seen.




### Special Orthogonal Group

The **Special Orthogonal Group**, written $SO(n)$, consists of:
- all $n \times n$ orthogonal matrices,
- with determinant $+1$, and
- the identity matrix.

The word "special" refers to this restriction to matrices with determinant $+1$. Matrices with determinant $-1$ (which include reflections and parity flips) are excluded.

Since orthogonal matrices preserve both vector lengths and angles, and requiring a determinant of $+1$ eliminates reflections, the matrices in $SO(n)$ represent **rotations** in $n$-dimensional space.



#### $SO(2)$

The group $SO(2)$ contains all **2D rotations**. A typical element of $SO(2)$ is the familiar $2 \times 2$ rotation matrix:

$$
\mathbf{R}(\theta) = 
\begin{bmatrix}
	\cos(\theta) & -\sin(\theta) \\
	\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

Here, $\theta$ is the counterclockwise rotation angle in the 2D plane.

The $SO(2)$ group is said to be **continuous**, since $\theta$ can vary continuously from $0$ to $2\pi$. This group describes systems with rotational symmetry in two dimensions, including rigid body motion, angular momentum in quantum mechanics, and some symmetric fields.




#### $SO(3)$

The group $SO(3)$ describes all **3D rotations**. Its elements include rotation matrices about the $x$-, $y$-, or $z$-axes, or any combination thereof. Like $SO(2)$, this group is continuous, with continuously varying angles.

$SO(3)$ is widely used in physics to describe rotational symmetries of three-dimensional systems, such as planetary motion, spin, or rotations of molecules and fields in space.




### Special Unitary Group

The **Special Unitary Group**, written $SU(n)$, is the set of all $n \times n$ complex matrices that are:
- **unitary**, and
- have determinant $+1$ (this is what "special" means in this naming convention).

A matrix is unitary if its inverse equals its **conjugate transpose**. This is also called the **Hermitian adjoint** (or simply the **adjoint** in physics). In symbols:

$$
\mathbf{A}^{\dagger} = (\mathbf{A}^*)^\text{T}
$$

Here, $\mathbf{A}^*$ means the complex conjugate of each element of $\mathbf{A}$, and the dagger $^{\dagger}$ means: take the complex conjugate, then transpose. So, a unitary matrix satisfies:

$$
\mathbf{A}^{-1} = \mathbf{A}^{\dagger} = (\mathbf{A}^*)^\text{T}
$$

Matrices in $SU(n)$ are of vital importance to quantum mechanics. They describe symmetries of quantum systems and represent quantum operations that preserve probability. The preservation os probability is key since the total probability must always be 1. 

In quantum mechanics, unitary matrices often appear as operators acting on quantum states. These include observable operators like position, momentum, and energy (via the Hamiltonian). For example, in **quantum computing**, the operations applied to qubits are typically elements of $SU(2)$, which governs the transformations of two-level quantum systems. Similarly, $SU(3)$ plays a vital role in high-energy physics: it is the symmetry group underlying Quantum Chromodynamics (QCD), which describes the strong force in the Standard Model of particle physics.










## Linear Dependence and Independence

In linear algebra, the ideas of **linear dependence** and **linear independence** are essential for understanding the structure of vector spaces. This includes simple coordinate systems (like $x$, $y$, $z$) and more advanced concepts like Hilbert Space, which shows up in quantum mechanics.

These concepts help us figure out whether a set of vectors can **span** a space. To span a space, the vectors you choose must 
1) be able to reproduce any other vector in that space using a linear combination 
	- Any vector possible in the space should be linearly dependent on the spanning set of vectors.
2) be linearly independent on each other
	- Any vector in the spanning set shoiuld not be linearly dependent on the other vectors in the spanning set.

For example, if you can write something like:

$$
a_1 \vec{v}_1 + a_2 \vec{v}_2 + a_3 \vec{v}_3 = a_4 \vec{v}_4
$$

and find nonzero values for $a_1$, $a_2$, and $a_3$ when $a_4$ is also nonzero, then $\vec{v}_4$ can be written using the other three vectors. That makes $\vec{v}_4$ **linearly dependent** on $\vec{v}_1$, $\vec{v}_2$, and $\vec{v}_3$.

If even one vector in a set can be written as a combination of the others, we say the set is **linearly dependent**. This means there is some redundancy. In the example above, if we can build $\vec{v}_4$ from the other three, then we do not really need it to span the space (that is, to fully cover it). It is not giving us a new direction or new information that we cannot already describe using the other vectors already present.

By contrast, a set is **linearly independent** when no vector in the set can be written as a combination of the others. Each one contributes a new and unique direction.

To test this mathematically, we can try to build the zero vector like this:

$$
a_1 \vec{v}_1 + a_2 \vec{v}_2 + a_3 \vec{v}_3 = \vec{0}
$$

If the only solution is $a_1 = a_2 = a_3 = 0$, then the vectors are independent. There is no way to write one using the others.

In matrix terms, this concept shows up when we use vectors as rows/columns. If the rows/columns of a matrix are linearly independent, then the matrix has a **non-zero determinant**. If they are dependent, the determinant is **zero**. This distinction is critical when solving systems of equations, determining whether a matrix has an inverse, and understanding transformations in physics.




## Determinants as a Tool to Test Linear Dependence

Let us connect this idea to something we have done before: **row reduction**.

Suppose we start with a $3 \times 3$ matrix:

$$
\mathbf{A} =
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

If Row 3 is linearly dependent on Rows 1 and 2, then we can write it as a combination of those rows. That means we could subtract that combination from Row 3 during row reduction and get all zeros:

$$
\mathbf{A}' =
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
0 & 0 & 0
\end{bmatrix}
$$

With a of zeros, we can readily see that the determinant is zero. This tells us the matrix is **singular**, which is just a fancy way of saying it does not have an inverse.

We only got a row of zeros because we assumed the rows/columns were dependent. So if a matrix has **linearly dependent rows/columns**, its **determinant is zero** and it **cannot be inverted**.

{% capture ex %}

If the determinant of a matrix is 0, then one or more of the rows/columns is linearly dependent on the others.

> **Consequence:** This matrix does **not have an inverse** because its determinant is zero!

{% endcapture %}
{% include result.html content=ex %}

So what is the takeaway?

> Build a matrix using the vectors as rows or columns.  
> - If the determinant is zero, the vectors are linearly dependent.  
> - If the determinant is not zero, the vectors are linearly independent.

Let us try this out with an example.

{% capture ex %}
Does the following matrix have an inverse?

$$
\begin{bmatrix}
	1 & 2 & 3 \\
	4 & 5 & 6 \\
	7 & 8 & 9 
\end{bmatrix}
$$

Let us check by calculating the determinant. If the determinant is zero, the matrix is **singular** and has **no inverse**.

We calculate the determinant using cofactor expansion along the first row:

$$
\begin{aligned}
	\begin{vmatrix}
		1 & 2 & 3 \\
		4 & 5 & 6 \\
		7 & 8 & 9 
	\end{vmatrix}
	&= 1 \begin{vmatrix}
	5 & 6 \\
	8 & 9
	\end{vmatrix}
	- 2 \begin{vmatrix}
	4 & 6 \\
	7 & 9
	\end{vmatrix}
	+ 3 \begin{vmatrix}
	4 & 5 \\
	7 & 8
	\end{vmatrix} \\
	&= 1(5 \cdot 9 - 6 \cdot 8)
	- 2(4 \cdot 9 - 6 \cdot 7)
	+ 3(4 \cdot 8 - 5 \cdot 7) \\
	&= 1(45 - 48)
	- 2(36 - 42)
	+ 3(32 - 35) \\
	&= -3 + 12 - 9 \\
	&= 0
\end{aligned}
$$

Since the determinant is zero, the matrix **does not have an inverse**. That means the rows (or columns) must be **linearly dependent**.

Let us test this by seeing whether one row can be built using the others. For instance, let's see if we can find constants $a$ and $b$ such that:

$$
a \cdot \text{Row 1} + b \cdot \text{Row 2} = \text{Row 3}
$$

That is:

$$
a \cdot \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} + b \cdot \begin{bmatrix} 4 & 5 & 6 \end{bmatrix} = \begin{bmatrix} 7 & 8 & 9 \end{bmatrix}
$$

This gives us a system of equations:

$$
\begin{aligned}
a + 4b &= 7\\
2a + 5b &= 8 \\
3a + 6b &= 9
\end{aligned}
$$

Let us solve the first equation for $a$:

$$
a = 7 - 4b
$$

Substitute into the second equation:

$$
\begin{aligned}
2(7 - 4b) + 5b &= 8 \\
14 - 8b + 5b &= 8 \\
14 - 3b &= 8 \\
-3b &= -6 \\
b &= 2
\end{aligned}
$$

Now plug back to find $a$:

$$
a = 7 - 4(2) = 7 - 8 = -1
$$

Let us check the third equation:

$$
3a + 6b = 3(-1) + 6(2) = -3 + 12 = 9
$$

which checks out. So we can write:

$$
-1 \cdot \text{Row 1} + 2 \cdot \text{Row 2} = \text{Row 3}
$$

This confirms that **Row 3 is linearly dependent** on the first two rows. 

Since one row can be written as a linear combination of the others:

- The rows/columns of the matrix are **linearly dependent** in some manner.
- The determinant is **zero**.
- The matrix has **no inverse**.

This isn't a formal proof, but is helps demonstrate the connections we are trying to make.
{% endcapture %}
{% include example.html content=ex %}







## Testing for Linear Dependence/Independence

In many cases, we need to determine whether a set of vectors is linearly dependent or independent. For instance, in 3-dimensional space, any vector can be expressed as a combination of three linearly independent vectors. We see this in practice with the commonly used vectors $ \hat{i} $, $ \hat{j} $, and $ \hat{k} $. These vectors form a set of **basis vectors** for 3-dimensional space, which are used to represent all other vectors in the space. While in general a space does not have a uniquely fixed set of basis vectors, certain conventions---like the use of $ \hat{i} $, $ \hat{j} $, and $ \hat{k} $ in Cartesian coordinates---are often followed to simplify calculations.

So, how do we determine if a set of vectors are linearly independent or not? One standard method is to arrange the vectors as columns of a matrix and compute the determinant. As we just determined in the section before this one, if the determinant of a matrix is 0 then the columns/rows of the matrix are not linearly independent.  

In more formal math language, we a saying a set of $ n $ vectors in $ n $-dimensional space are linearly independent if and only if the determinant of matrix $ \mathbf{A} $ is non-zero, where matrix $ \mathbf{A} $ is an $ n \times n $ matrix whose columns are made up of the vectors being tested: $ \vec{v}_1, \vec{v}_2, \dots, \vec{v}_n $. To be explicit, matrix $ \mathbf{A} $ can be written in the following manner

$$
\mathbf{A} = 
\begin{bmatrix}
	\vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_n
\end{bmatrix}
$$

The determinant of $ \mathbf{A} $, satisfies the condition:
$$
\det(\mathbf{A}) \neq 0 \quad \text{(linearly independent)}.
$$
If $ \det(\mathbf{A}) = 0 $, then the vectors are linearly dependent, meaning that at least one vector can be expressed as a linear combination of the others as per the discussion in previous sections.

{% capture ex %}

As an example, consider three vectors in 3-dimensional space:

$$
\vec{v}_1 = 
\begin{bmatrix}
	1 \\ 0 \\ 0
\end{bmatrix} \quad
\vec{v}_2 = 
\begin{bmatrix}
	0 \\ 1 \\ 0
\end{bmatrix}, \quad
\vec{v}_3 = 
\begin{bmatrix}
	1 \\ 1 \\ 0
\end{bmatrix}
$$

We form the matrix:

$$
\mathbf{A} = 
\begin{bmatrix}
	1 & 0 & 1 \\
	0 & 1 & 1 \\
	0 & 0 & 0
\end{bmatrix}
$$

The determinant of $ \mathbf{A} $ is:

$$
\text{det}(\mathbf{A}) = 1 \cdot 
\begin{vmatrix}
	1 & 1 \\
	0 & 0
\end{vmatrix} - 0 \cdot 
\begin{vmatrix}
	0 & 1 \\
	0 & 0
\end{vmatrix} + 1 \cdot 
\begin{vmatrix}
	0 & 1 \\
	0 & 0
\end{vmatrix} = 0
$$

Since $ \det(\mathbf{A}) = 0 $, the vectors are linearly dependent. We can easily see that $ \vec{v}_3 = \vec{v}_1 + \vec{v}_2 $.

{% endcapture %}
{% include example.html content=ex %}

This process generalizes to higher dimensions and any number of vectors, making it a powerful tool for verifying linear independence.

















## Rank of a Matrix

The **rank of a matrix** is the maximum number of linearly independent rows or columns in the matrix. In physics, the rank often represents the number of independent equations in a system. For example, conservation laws—such as the conservation of momentum or mechanical energy—introduce linear dependencies among variables. This results in a reduction in the rank, indicating fewer available degrees of freedom. 

As we can see, the rank of a matrix is closely tied to the concept of **linear dependence**:
- If the rank of the matrix is **less than the total number of rows (or columns)**, then the rows (or columns) are linearly dependent.
- If the rank equals the number of rows (or columns), they are linearly independent.


For nonsquare matrices, such as an $m \times n$ matrix, the rank is always less than or equal to the $m$ or $n$, whichever is smaller. In other words, a matrix cannot have a rank greater than the smaller of its two dimensions (number of rows or columns).


{% capture ex %}

Suppose a system of three particles, each moving in one spatial dimension, satisfies a conservation law for total momentum -- the total momentum of the three particle system is zero. If the momentum of each particle in one dimension is represented as $p_1$, $p_2$, and $p_3$, then the conservation law can be expressed as:

$$
p_1 + p_2 + p_3 = 0
$$

We can turn this equation into a matrix equation in the following manner: 

$$
\mathbf{A} \vec{p} = \vec{0}
$$

where:

$$
\mathbf{A} = \begin{bmatrix}
	1 & 1 & 1
\end{bmatrix} \qquad\qquad
\vec{p} = \begin{bmatrix}
	p_1 \\ p_2 \\ p_3
\end{bmatrix} \qquad\qquad
\vec{0} = \begin{bmatrix}
	0
\end{bmatrix}
$$

Notice matrix $\mathbf{A}$ has a single row, this suggests that the conservation law is a constraint on the system -- the sum of the momenta will be zero

Conservation laws in physics often correspond to constraints represented by linearly dependent rows or columns in a matrix -- though we do not generally write them in this manner. For example, in classical mechanics, the conservation of mechanical energy, momentum, or angular momentum can be expressed using similar matrix equations to the one we wrote above.

In this example, the matrix $\mathbf{A}$ enforces the conservation law. A rank analysis of $\mathbf{A}$ shows that the system has two degrees of freedom (since the rank of $\mathbf{A}$ is 1, leaving $3 - 1 = 2$ free variables). This aligns with our physical intuition: two momenta can be independently specified, but the third is constrained by the conservation law.

{% endcapture %}
{% include example.html content=ex %}








## Conceptual Example of these Concepts in Practice

**Conservation of Angular Momentum in Rotating Systems**

Imagine a rigid body rotating in three dimensions. The state of the system can be described by its angular velocity vector $\vec{\omega}$ and moment of inertia tensor $\mathbf{I}$. The relationship between the angular momentum $\vec{L}$ and the angular velocity is given by:
$$
\vec{L} = \mathbf{I}  \vec{\omega}
$$
This equation is similar in concept for linear momentum  $\vec{p} = m \vec{v}$, but with the angular velocity $\vec{omega}$ in place of the linear velocity $\vec{v}$ and  the momentum of inertia tensor (a matrix)$\mathbf{I}$ in place of the mass $m$.


**Orthogonal Matrices in Rotational Transformations**\\
Suppose we want to transform the system into a new coordinate frame that is rotated by some angle. This transformation is represented by an orthogonal matrix $\mathbf{R}$:
$$
\vec{\omega}' = \mathbf{R}  \vec{\omega} \quad \text{and} \quad \vec{L}' = \mathbf{R}  \vec{L}
$$
The orthogonality of $\mathbf{R}$ ensures that the magnitudes of $\vec{\omega}$ and $\vec{L}$ are preserved, as well as the angles between them. This is crucial for preserving physical laws like conservation of angular momentum.

**Linear Independence and the Moment of Inertia Tensor**\\
The moment of inertia tensor $\mathbf{I}$ is a $3 \times 3$ matrix when working in 3 spacial dimensions. If $\mathbf{I}$ has full rank (rank = 3), then the angular velocity components along the three spacial axes are linearly independent. This means that motion along one axis does not depend on the others. 

However, if $\mathbf{I}$ loses rank (say, due to symmetry in the geometry of the rotating object or maybe due to a constraint added to the system. For instance, such as a thin rod rotating about its longitudinal axis would be an example of the rotating object having a geometric symmetry that will cause a reduction in the rank of the matrix. As a result, certain components of $\vec{\omega}$ will become linearly dependent, reducing the system's degrees of freedom.

**Rank and Conservation Laws**\\
Now consider a scenario where the angular momentum vector $\vec{L}$ is conserved. This will act as a constrain on the system, which can be written in a general manner as:
$$
\vec{L} \cdot \hat{n} = \text{constant} \quad \text{(e.g., for rotation about a fixed axis $\hat{n}$)}
$$
This constraint introduces a linear dependency among the components of $\vec{L}$, which effectively reduces the rank of the system's equations. 

**Determinants and Volume Preservation**\\
The determinant of $\mathbf{R}$ (which must be $\pm 1$ since it is an orthogonal matrix) ensures that the transformation preserves the ``volume" of the angular velocity space. If the determinant were zero, it would mean a collapse of dimensionality, indicating a loss of degrees of freedom, which is physically inadmissible for a rotational transformation.























## Application:

Let’s consider a rigid body in 3D space. Suppose the body is initially at rest in a configuration where its velocity components along the $ x $-, $ y $-, and $ z $-axes are linearly independent. We will then rotate this body, and we will see how linear dependence of the vectors changes under rotation -- it shouldn't.

**Initial Linear Independence of Velocity Components**

Let the velocity of the rigid body be given by three velocity vectors representing the motion of different points in the body along the $ x $-, $ y $-, and $ z $-axes, respectively:

$$
\vec{v_1} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} \qquad \qquad \qquad 
\vec{v_2} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \quad \qquad \qquad 
\vec{v_3} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

These vectors are clearly linearly independent because no vector can be written as a linear combination of the other two. We can check this by forming a matrix with the vectors, letting each vector represent a column in the matrix, and then calculating the determinant: 

$$
\text{det} \begin{bmatrix} \vec{v_1} & \vec{v_2} & \vec{v_3} \end{bmatrix} = \text{det} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = 1 \neq 0
$$

Since the determinant is non-zero, the vectors are linearly independent, meaning there are no constrains to the motion of the object at play and there is full spatial freedom for the body’s motion in 3D space.

**Rotation of the Rigid Body Using an Orthogonal Matrix**

Now, let’s apply a rotation to this system. Consider a rotation matrix $ \mathbf{R} $ that rotates the body by $ 90^\circ $ around the $ z $-axis:

$$
\mathbf{R} = \begin{bmatrix} 
	\cos(90^\circ) & -\sin(90^\circ) & 0 \\
	\sin(90^\circ) & \cos(90^\circ) & 0 \\
	0 & 0 & 1 
\end{bmatrix} 
= \begin{bmatrix} 
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{bmatrix}
$$

When we apply this matrix to the velocity vectors, we get the new velocity components:

$$
\mathbf{R} \vec{v_1} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \quad \qquad \qquad 
\mathbf{R} \vec{v_2} = \begin{bmatrix} -1 \\ 0 \\ 0 \end{bmatrix} \quad \qquad \qquad 
\mathbf{R} \vec{v_3} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

**Checking Linear Independence After Rotation**

After applying the rotation, the new vectors are:

$$
\vec{v_1'} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \quad \qquad \qquad 
\vec{v_2'} = \begin{bmatrix} -1 \\ 0 \\ 0 \end{bmatrix} \quad \qquad \qquad 
\vec{v_3'} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

We now check whether these vectors are still linearly independent. To do this, we compute the determinant of the matrix formed by these vectors as columns:

$$
\text{det} \begin{bmatrix} \vec{v_1'} & \vec{v_2'} & \vec{v_3'} \end{bmatrix} = \text{det} \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix} = 1
$$

Since the determinant is non-zero, the vectors are still linearly independent after the rotation. This demonstrates that rotations do not change the linear independence of the vectors. Essentially what this means is the space began as a 3-dimensional space and remained a 3-dimensional space after the rotation.

**Introducing Linear Dependence**

Now, let’s consider a scenario where we add a constraint to the system, such as a **conservation law** like conservation of energy or angular momentum. If the body is constrained to only move around the $xy$-plane. Let’s modify the velocity vectors to reflect this constraint:

$$
\vec{v_1} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} \quad \qquad \qquad 
\vec{v_2} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \quad \qquad \qquad 
\vec{v_3} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

Now, the matrix formed by these vectors has a determinant of zero:

$$
\text{det} \begin{bmatrix} \vec{v_1} & \vec{v_2} & \vec{v_3} \end{bmatrix} = \text{det} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix} = 0
$$

Since the determinant is zero, the velocity vectors are now **linearly dependent**. This reflects that the system is constrained, and there are fewer degrees of freedom in the motion of the rigid body (in this case, only motion in the $ x $- and $ y $-plane is possible due to conservation of angular momentum about the $ z $-axis).















## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.



Consider a rigid body in 3D space with velocity components along the $x$-, $y$-, and $z$-axes represented by the following velocity vectors:

$$
\vec{v_1} = \begin{bmatrix} 2 \\ 0 \\ 0 \end{bmatrix}, \quad 
\vec{v_2} = \begin{bmatrix} 0 \\ 3 \\ 0 \end{bmatrix}, \quad 
\vec{v_3} = \begin{bmatrix} 0 \\ 0 \\ 4 \end{bmatrix}
$$


a) Prove that these velocity vectors are initially linearly independent.  

b) A rotation is applied to the rigid body using the following orthogonal rotation matrix $ \mathbf{R} $ which represents a $ 90^\circ $ rotation about the $z$-axis:

$$
\mathbf{R}_z(90^\circ) = \begin{bmatrix} 
	\cos(90^\circ) & -\sin(90^\circ) & 0 \\
	\sin(90^\circ) & \cos(90^\circ) & 0 \\
	0 & 0 & 1 
\end{bmatrix} 
= \begin{bmatrix} 
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{bmatrix}
$$

Use the matrix $ \mathbf{R}_z(90^\circ) $ to find the new rotated velocity vectors $ \vec{v_1} $, $ \vec{v_2} $, and $ \vec{v_3} $.  

c) After applying the rotation, check whether the velocity vectors are still linearly independent.  

d) Now, assume a constraint on the motion forcing the system to only move about in the $yz$-plane. This results in the velocity vectors being updated as follows:

$$
\vec{v_1} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \quad 
\vec{v_2} = \begin{bmatrix} 0 \\ 3 \\ 0 \end{bmatrix}, \quad 
\vec{v_3} = \begin{bmatrix} 0 \\ 0 \\ 4 \end{bmatrix}
$$

Check the linear dependence of the new velocity vectors. What is the rank of the matrix (how many columns/rows are linearly independent) formed by the new vectors? 







