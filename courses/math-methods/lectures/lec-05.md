---
layout: default
title: Mathematical Methods - Lecture 05
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 5
---


# Lecture 05 – Other Special Matrices and Orthogonal Matrices Part 1


This lecture is going to be one part review, one part fitting in new theory that is useful for those who are interested in doing theoretical physics in the future, and one part introduction of a new concept. 

Let's begin with the review. Specifically, reminding ourselves about using matrices as operators, like rotations, reflections, and scaling.


## Review of Matrices as Coordinate Transformations/Operators

First, let’s review what we know about matrices when they are used to represent coordinate transformations. Matrices used in this context are often referred to as *operators*. However, it’s important to note that the term *operator* is much broader than this. 

In general, an operator can describe many different kinds of transformations, not just linear coordinate transformations that can be represented by matrices. At its core, an operator is a mathematical object that has been defined to *do something*. That is, to act on another object and produce a result. The object being acted on might be a number, a function, a vector, or something more abstract.

To give us something to dig our teeth into, let consider a simple operator. Let's define an “add 1” operator, whose sole job is to do what it says on the tin: add 1 to whatever it acts on. We could represent this operator symbolically as:

$$
A_{+1} = (1 + )
$$

When we apply this operator to a number, we get results like:

$$
A_{+1}\, 4 = (1 + )\,4 = 1 + 4 = 5
$$

or, if we apply it to a function,

$$
A_{+1}\, x^2 = (1 + )\,x^2 = 1 + x^2 = x^2 + 1.
$$

In each case, the operator performs the action it was designed to acheive, in this case adding 1, regardless of what it acts on.

Another operator most of us are already very familiar with is the derivative. The first derivative with respect to $x$ can be written explicitly as an operator in the following manner:

$$
D_x = \frac{d}{dx}
$$

When this operator acts on a function, it returns the derivative of that function. For example,

$$
D_x (x^2 + 1) = \frac{d}{dx} (x^2 + 1) = 2x
$$

The takeaway here is that we already work with operators all the time. We just don’t always call them that or stop to think about them in this abstract way. In this lecture, we’ll build on this familiar idea and focus on particular classes of operators that can be represented by matrices and explore the special properties they have.






### Reflection Matrices

Reflections across the $ x $-, $ y $-, and $ z $-axes in three dimensions can be represented by the following matrices. 

$$
\mathbf{R}_x = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & -1 & 0 \\
	0 & 0 & -1 
\end{bmatrix}
\qquad 
\mathbf{R}_y = \begin{bmatrix}
	-1 & 0 & 0 \\
	0 & 1 & 0 \\
	0 & 0 & -1 
\end{bmatrix}
\qquad
\mathbf{R}_z = \begin{bmatrix}
	-1 & 0 & 0 \\
	0 & -1 & 0 \\
	0 & 0 & 1 
\end{bmatrix}
$$

Each matrix corresponds to a reflection across one of the major axes:

- $ \mathbf{R}_x $: Reflects across the $ x $-axis by flipping the signs of the $ y $- and $ z $-coordinates.
- $ \mathbf{R}_y $: Reflects across the $ y $-axis by flipping the signs of the $ x $- and $ z $-coordinates.
- $ \mathbf{R}_z $: Reflects across the $ z $-axis by flipping the signs of the $ x $- and $ y $-coordinates.


These matrices flip the sign of the coordinates along the respective axis while leaving the other coordinates unchanged. These reflection matrices are useful for identifying symmetries in physical systems, which can lead one to conservation laws, as we will see later.



### Matrices as Operators and Coordinate Transformations


Recall that to apply a matrix transformation to a vector written as a **column vector**, we multiply the transformation matrix on the **left**. The result is a new column vector:

$$
\mathbf{T} \vec{r} = \vec{r}' \implies \mathbf{T} \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} a' \\ b' \\ c' \end{bmatrix}
$$

This convention is the most common in physics and engineering, where vectors are typically represented as column vectors and matrices act on them from the left.

To transform a **matrix** that represents a physical quantity under a change of coordinates or basis, the transformation must act on **both sides** of the matrix. For a general transformation, this is done by multiplying on the left by the transformation matrix and on the right by its inverse:

$$
\mathbf{A}' = \mathbf{T}\,\mathbf{A}\,\mathbf{T}^{-1}
$$

This form can be simplified a bit when the transformation $ \mathbf{T} $ represents a rotation or reflection, since these matrices are orthogonal which satisfy $ \mathbf{T}^{-1} = \mathbf{T}^\text{T} $. We will discuss orthogonal matrices in more detail below.







## Some other Special Matrices and Operations

### Identity Matrix

### Identity Matrix

We’ve already seen this matrix, but now is a good time to take a closer look at its properties. The **identity matrix**, usually denoted by $ \mathbf{I} $ or sometimes $ \mathbf{1} $, is a square ($n \times n$) matrix that acts as the multiplicative identity in matrix operations. That means for any compatible matrix $ \mathbf{A} $, multiplying by the identity on either side leaves $ \mathbf{A} $ unchanged:

$$
\mathbf{I} \, \mathbf{A} = \mathbf{A} \, \mathbf{I} = \mathbf{A}
$$

This is analogous to multiplying a number by 1:

$$
1 \cdot 3 = 3 \cdot 1 = 3
$$

You can think of the identity matrix as something like the matrix version of the number 1.

Using this definition, we can also see how it relates to matrix inverses. If a matrix $ \mathbf{A} $ has an inverse, then:

$$
\mathbf{A} \, \mathbf{A}^{-1} = \mathbf{I}
\qquad \text{and} \qquad
\mathbf{A}^{-1} \, \mathbf{A} = \mathbf{I}
$$

So the identity matrix naturally appears as the result of multiplying a matrix by its inverse, the side doesn't matter.

When writting it out, the identity matrix is defined as having ones on the main diagonal and zeros everywhere else. A general $ n \times n $ identity matrix looks like this:

$$
\mathbf{I}_n = 
\begin{bmatrix}
	1 & 0 & 0 & \cdots & 0 \\
	0 & 1 & 0 & \cdots & 0 \\
	0 & 0 & 1 & \cdots & 0 \\
	\vdots & \vdots & \vdots & \ddots & \vdots \\
	0 & 0 & 0 & \cdots & 1
\end{bmatrix}
$$

In geometric terms, the identity matrix represents a **“do-nothing” transformation**: applying it to a vector leaves the vector unchanged. 







### Diagonal Matrix

A **diagonal matrix** is a square ($n\times n$) matrix in which all the entries outside the main diagonal are zero.  A general $ n \times n $ diagonal matrix can be written as:

$$
\mathbf{D} = 
\begin{bmatrix}
	d_1 & 0 & 0 & \cdots & 0 \\
	0 & d_2 & 0 & \cdots & 0 \\
	0 & 0 & d_3 & \cdots & 0 \\
	\vdots & \vdots & \vdots & \ddots & \vdots \\
	0 & 0 & 0 & \cdots & d_n
\end{bmatrix}
$$

Diagonal matrices are important because they are extremely simple to work with, especially in operations like matrix multiplication or finding powers of matrices (multiple matrix multiplications). For example, consider a $2 \times 2$ diagonal matrix:

$$
\mathbf{D} = 
\begin{bmatrix}
	d_1 & 0  \\
	0 & d_2 
\end{bmatrix}
$$

and multiple it with itself:

$$
\mathbf{D}^2 = \mathbf{D}\mathbf{D} = 
\begin{bmatrix}
	d_1 & 0  \\
	0 & d_2 
\end{bmatrix} \begin{bmatrix}
	d_1 & 0  \\
	0 & d_2 
\end{bmatrix} = \begin{bmatrix}
	d_1^2 & 0  \\
	0 & d_2^2 
\end{bmatrix} \quad\implies\quad \mathbf{D}^2 = \begin{bmatrix}
	d_1^2 & 0  \\
	0 & d_2^2 
\end{bmatrix} 
$$

By extending this to multiple multiplications, you can show:

$$
\mathbf{D}^k = \begin{bmatrix}
	d_1^k & 0  \\
	0 & d_2^k 
\end{bmatrix} 
$$

and in general:

$$
\mathbf{D}^k = \underbrace{\mathbf{D}\mathbf{D}\cdots\mathbf{D}}_{\text{$k$ times}} =
\begin{bmatrix}
	d_1^k & 0 & 0 & \cdots & 0 \\
	0 & d_2^k & 0 & \cdots & 0 \\
	0 & 0 & d_3^k & \cdots & 0 \\
	\vdots & \vdots & \vdots & \ddots & \vdots \\
	0 & 0 & 0 & \cdots & d_n^k
\end{bmatrix}.
$$

Diagonal matrices often appear in systems with independent components, such as the principal axes in rotational systems or in eigenvalue problems where a matrix is diagonalized (a matrix is rewritten in a diagonal manner), which we will cover in a future lecture. In many cases, when working with a system described by non-diagonal matrices, our first step is generally to find a coordinate system or representation that allows all relevant matrices to be expressed in a diagonal form, if possible.






### Symmetric Matrix

A **symmetric matrix** is a square ($n\times n$) matrix that is equal to its own transpose. That is, a matrix $ \mathbf{A} $ is symmetric if,

$$ \mathbf{A}^T = \mathbf{A} $$ 

This property means that the elements across the main diagonal are symmetric; that is, the element in the $ i $-th row and $ j $-th column is the same as the element in the $ j $-th row and $ i $-th column:

$$
a_{ij} = a_{ji}
$$

For example, a symmetric matrix might look like:

$$
\mathbf{A} = 
\begin{bmatrix}
	a_{11} & {\color{red}a_{12}} & {\color{blue}a_{13}} \\
	{\color{red}a_{12}} & a_{22} & {\color{purple}a_{23}} \\
	{\color{blue}a_{13}} & {\color{purple}a_{23}} & a_{33}
\end{bmatrix}
$$

notice the off-diagonal elements are equal in a symmetric manner. 

Symmetric matrices often arise in systems that exhibit some form of symmetry. For example, in the mechanics of materials and elasticity theory, the stress-strain tensor is symmetric since to the strain along the $xy$-plane is the same as along the $yx$-plane. Similarly, in general relativity, the stress-energy tensor is also symmetric for similar reasons.




### Antisymmetric Matrix

An **antisymmetric matrix** (or skew-symmetric matrix) is a square ($n\times n$)  matrix that satisfies the property

$$ \mathbf{A}^T = -\mathbf{A} $$ 

This means that for every element in the matrix, the value of the element at the $ i $-th row and $ j $-th column is the negative of the element at the $ j $-th row and $ i $-th column:

$$
a_{ij} = -a_{ji}
$$

Notice what happens if we take this definition and apply it to the main diagonal:

$$
a_{ii} = -a_{ii}
$$

which is only possible if $a_{ii} = 0$. This implies all the diagonal elements of an antisymmetric matrix are zero. 

So, a $ 3 \times 3 $ antisymmetric matrix might look like:

$$
\mathbf{A} = 
\begin{bmatrix}
	0 & {\color{red}a_{12}} & {\color{blue}a_{13}} \\
	-{\color{red}a_{12}} & 0 & {\color{purple}a_{23}} \\
	-{\color{blue}a_{13}} & -{\color{purple}a_{23}} & 0
\end{bmatrix}
$$

Antisymmetric matrices are commonly found in the study of angular momentum, where the components of the angular momentum operator form an antisymmetric matrix resulting from the cross product. In fact, the cross product of a vector acting on another vector can be written as a matrix operator. 




#### Cross Product Matrix

As we have discussed previously, the cross product of two vectors $ \vec{A} $ and $ \vec{B} $ is a vector that is perpendicular to both $ \vec{A} $ and $ \vec{B} $, with magnitude equal to the area of the parallelogram formed by the vectors. We may also recall that the cross product in 3-dimensional space can be calculated using the determinant method in the following manner:

$$
\mathbf{A} \times \mathbf{B} = \begin{vmatrix} 
	\hat{i} & \hat{j} & \hat{k} \\
	A_x & A_y & A_z \\
	B_x & B_y & B_z
\end{vmatrix}
$$

A convenient way to compute the cross product, especially in matrix form, is using a special type of antisymmetric matrix called the cross-product matrix. 

The cross-product matrix $ [\vec{A}]_\times $ represents the cross product operation with vector $ \vec{A} $ being on the left. That is,

$$
\vec{A} \times \vec{B} = [\vec{A}]_\times \, \vec{B}
$$

The matrix $ [\vec{A}]_\times $ is defined in the following manner:

$$
[\vec{A}]_\times = \begin{bmatrix}
	0 & -A_z & A_y \\
	A_z & 0 & -A_x \\
	-A_y & A_x & 0
\end{bmatrix}
$$

When this matrix acts on a column vector, say:

$$ \vec{B} = \begin{bmatrix} B_x \\ B_y \\ B_z \end{bmatrix} $$

the result is the cross product between the two matrices:

$$
\vec{A} \times \vec{B} = [\vec{A}]_\times \vec{B} =  \begin{bmatrix}
	0 & -A_z & A_y \\
	A_z & 0 & -A_x \\
	-A_y & A_x & 0
\end{bmatrix} \begin{bmatrix} B_x \\ B_y \\ B_z \end{bmatrix} = \begin{bmatrix}
A_y B_z - A_z B_y \\
A_z B_x - A_x B_z \\
A_x B_y - A_y B_x
\end{bmatrix}
$$

Let's look at an example of this in action

{% capture ex %}
Let $ \vec{A} = (1, 2, 3) $ and $ \vec{B} = (4, 5, 6) $. We want to compute the cross product $ \vec{A} \times \vec{B} $ using the cross product matrix. 

First, we write the cross product matrix for $\vec{A}$:

$$
[\vec{A}]_\times = \begin{bmatrix}
   0 & -3 & 2 \\
   3 & 0 & -1 \\
   -2 & 1 & 0
\end{bmatrix}
$$

Now we compute the cross product by acting on $\vec{B}$ with this matrix:

$$
\begin{aligned}
\vec{A} \times \vec{B} &= [\vec{A}]_\times \vec{B} \\
&= \begin{bmatrix}
   0 & -3 & 2 \\
   3 & 0 & -1 \\
   -2 & 1 & 0
\end{bmatrix}
\begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}\\
 &= \begin{bmatrix}
   -15 + 12 \\
   12 - 6 \\
   -8 + 5
\end{bmatrix}\\
&= \begin{bmatrix}
   -3 \\
   6 \\
   -3
\end{bmatrix} \\
\vec{A} \times \vec{B} &= \begin{bmatrix}
-3 \\
6 \\
-3
\end{bmatrix}
\end{aligned}
$$

{% endcapture %}
{% include example.html content=ex %}



{% capture ex %}
Suppose we have a charged particle, charge $q$, traveling with a velocity of $\vec{v}$ in an external electric field $\vec{E}$ and magnetic field $\vec{B}$. These vectors can be given as:

$$ \vec{v} = \begin{bmatrix}
   v_x \\ v_y \\ v_z
\end{bmatrix} \qquad \vec{E} = \begin{bmatrix}
E_x \\ E_y \\ E_z
\end{bmatrix} \qquad \vec{B} = \begin{bmatrix}
B_x \\ B_y \\ B_z
\end{bmatrix} \qquad $$

The electromagnetic force equation can be calculated via the Lorentz force equation:

$$ \vec{F}^{EM} = q\left( \vec{E} + \vec{v} \times \vec{B} \right) $$

We can use the cross product matrix for the velocity to get:

$$
\begin{aligned}
\vec{F}^{EM} &= q\left( \begin{bmatrix}
   E_x \\ E_y \\ E_z
\end{bmatrix} + \begin{bmatrix}
0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0
\end{bmatrix} \begin{bmatrix}
B_x \\ B_y \\ B_z
\end{bmatrix} \right) \\
&= q\left( \begin{bmatrix}
E_x \\ E_y \\ E_z
\end{bmatrix} +  \begin{bmatrix}
-v_z B_y + v_y B_z \\ v_z B_x - v_x B_z \\ -v_y B_x + v_x B_y
\end{bmatrix} \right)\\
\vec{F}^{EM} &=  q\begin{bmatrix}
E_x + v_y B_z-v_z B_y \\ E_y + v_z B_x - v_x B_z \\ E_z + v_x B_y-v_y B_x
\end{bmatrix}
\end{aligned}
$$

{% endcapture %}
{% include example.html content=ex %}


The cross-product matrix formulation helps simplifies calculations, especially when working with angular momentum, torque, and rotational dynamics, where the cross product frequently appears. Further, this matrix is of vital importance to electricity and magnetism where you can unify the electric and magnetic fields into the single $ 4 \times 4 $ matrix:

$$ 
\mathbf{F} = \begin{bmatrix}
0 & E_x/c & E_y/c & E_z/c \\
-E_x/c &  0 & - B_z & B_y \\
-E_y/c &  B_z & 0 & - B_x \\
-E_x/c &  - B_y & B_x & 0 
\end{bmatrix} 
$$

This is an anisymmetric matrix, and notice the part with the magnetic field is $[\vec{B}]_\times$. The Maxwell equations then become manipulations to this matrix.








### Representing a Matrix as Symmetric and Antisymmetric Components

It turns out any square ($n\times n$) matrix can be decomposed into its symmetric and antisymmetric components. This is useful in various applications in physics and engineering, where symmetric and antisymmetric matrices often represent different physical phenomena, such as the stress tensor (symmetric) or the angular momentum operator (antisymmetric).

Given a square matrix $ \mathbf{A} $, we can decompose it into a symmetric matrix $ \mathbf{A}_S $ and an antisymmetric matrix $ \mathbf{A}_A $ using some intentional addition and subtractions. To being, let's assume we can make the following decomposition of $\mathbf{A}$:

$$ \mathbf{A} = \mathbf{A}_S + \mathbf{A}_A $$

We can take the transpose of this, and leverage the symmetric and antisymmetric nature of these components to get:

$$ \mathbf{A}^\text{T} = \mathbf{A}_S^\text{T} + \mathbf{A}_A^\text{T} \quad\implies\quad \mathbf{A}^\text{T} = \mathbf{A}_S - \mathbf{A}_A $$

Adding $\mathbf{A}$ and $\mathbf{A}^\text{T}$ gives:

$$ \mathbf{A} + \mathbf{A}^\text{T} = 2 \mathbf{A}_S \quad\implies\quad \mathbf{A}_S = \frac{1}{2} \left( \mathbf{A} + \mathbf{A}^\text{T} \right)  $$

and similarly subtracting gives:

$$ \mathbf{A} - \mathbf{A}^\text{T} = 2 \mathbf{A}_A \quad\implies\quad \mathbf{A}_A = \frac{1}{2} \left( \mathbf{A} - \mathbf{A}^\text{T} \right)  $$

This means a matrix, written in terms of these components, can be written as:

$$ \mathbf{A} = \mathbf{A}_S + \mathbf{A}_A = \frac{1}{2} \left( \mathbf{A} + \mathbf{A}^\text{T} \right) + \frac{1}{2} \left( \mathbf{A} - \mathbf{A}^\text{T} \right)  $$


{% capture ex %}
For example, consider the matrix:

$$
\mathbf{A} = 
\begin{bmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
\end{bmatrix}
$$

The symmetric and antisymmetric components of $ \mathbf{A} $ are:

$$
\mathbf{A}_S = \frac{1}{2} \left(
\begin{bmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
\end{bmatrix} +
\begin{bmatrix}
   1 & 4 & 7 \\
   2 & 5 & 8 \\
   3 & 6 & 9
\end{bmatrix}\right)
= \begin{bmatrix}
   1 & 3 & 5 \\
   3 & 5 & 7 \\
   5 & 7 & 9
\end{bmatrix}
$$

$$
\mathbf{A}_A = \frac{1}{2} \left(
\begin{bmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
\end{bmatrix} - 
\begin{bmatrix}
   1 & 4 & 7 \\
   2 & 5 & 8 \\
   3 & 6 & 9
\end{bmatrix} \right)
= \begin{bmatrix}
   0 & 1 & 2 \\
   -1 & 0 & 3 \\
   -2 & -3 & 0
\end{bmatrix}
$$
{% endcapture %}
{% include example.html content=ex %}











## Orthogonal Matrices

A classification of matrices we have encountered a couple of times already, but have not yet explored in detail is the **orthogonal matrix**. A matrix $\mathbf{A}$ is said to be orthogonal if its inverse $\mathbf{A}^{-1}$ is equal to its transpose $ \mathbf{A}^\text{T}$:

$$
\mathbf{A}^{-1} = \mathbf{A}^\text{T}
$$

Why are these called *orthogonal* matrices? To understand this, let's think of a matrix as a collection of column vectors arranged in sequence to form the matrix:

$$
\mathbf{A} = \begin{bmatrix}
	| &| &|  \\ \vec{a}_1 & \vec{a}_2 & \vec{a}_3 \\ | &| &|  
\end{bmatrix}   = \begin{bmatrix}
	\begin{bmatrix} a_{x1} \\ a_{y1} \\ a_{z1} \end{bmatrix} & 
	\begin{bmatrix} a_{x2} \\ a_{y2} \\ a_{z2} \end{bmatrix} & 
	\begin{bmatrix} a_{x3} \\ a_{y3} \\ a_{z3} \end{bmatrix}
\end{bmatrix} = \begin{bmatrix}
	a_{x1} & a_{x2} & a_{x3} \\
	a_{y1} & a_{y2} & a_{y3} \\
	a_{z1} & a_{z2} & a_{z3}
\end{bmatrix}
$$

where we have intentionally written the subscripts so that they will agree with the $a_{\text{row, column}}$ (rail-car; row then column) format we have adopted for matrices. 

Taking the transpose of this matrix gives:

$$ \mathbf{A}^\text{T} =  \begin{bmatrix}
   a_{x1} & a_{y1} & a_{z1} \\[1.15ex]
   a_{x2} & a_{y2} & a_{z2} \\[1.15ex]
   a_{x3} & a_{y3} & a_{z3}
\end{bmatrix} = \begin{bmatrix}
   \begin{bmatrix}
      a_{x1} & a_{y1} & a_{z1}
   \end{bmatrix} \\[1.15ex]\begin{bmatrix}
      a_{x2} & a_{y2} & a_{z2}
   \end{bmatrix} \\[1.15ex]\begin{bmatrix}
      a_{x3} & a_{y3} & a_{z3}
   \end{bmatrix} \\
\end{bmatrix} = \begin{bmatrix}
   \vec{a}_1^\text{T} \\[1.15ex] \vec{a}_2^\text{T} \\[1.15ex] \vec{a}_3^\text{T}
\end{bmatrix}  $$

where $\vec{a}_1^\text{T}$, $\vec{a}_2^\text{T}$, and  $\vec{a}_3^\text{T}$ are row vectors (transposed column vectors).

Now, let's take $\mathbf{A}$ and operate on the left with $\mathbf{A}^\text{T}$:

$$  \mathbf{A}^\text{T} \mathbf{A}  =  \begin{bmatrix}
	\vec{a}_1^\text{T} \\ \vec{a}_2^\text{T} \\ \vec{a}_3^\text{T}
\end{bmatrix} \left[ \frac{}{} \vec{a}_1 \,\,\, \vec{a}_2 \,\,\, \vec{a}_3       \frac{}{} \right]  = \begin{bmatrix}
	\vec{a}_1^\text{T}  \vec{a}_1& \vec{a}_1^\text{T}  \vec{a}_2 & \vec{a}_1^\text{T} \vec{a}_3  \\
	\vec{a}_2^\text{T}  \vec{a}_1& \vec{a}_2^\text{T}  \vec{a}_2 & \vec{a}_2^\text{T} \vec{a}_3  \\
	\vec{a}_3^\text{T}  \vec{a}_1& \vec{a}_3^\text{T}  \vec{a}_2 & \vec{a}_3^\text{T} \vec{a}_3
\end{bmatrix}  $$

Notice we have row vectors $\vec{a}^\text{T}$ acting from the left on a column vectors $\vec{a}$. But, that is just a vector dot product:

$$ \vec{a}^\text{T}\vec{b} = \begin{bmatrix}
	a_1 & a_2 & a_3 
\end{bmatrix}  \begin{bmatrix}
	b_1 \\ b_2 \\ b_3 
\end{bmatrix} = a_1 b_1 + a_2 b_2 + a_3 b_3  =  \vec{a} \cdot \vec{b}  $$

This realization allows us to rewrite the previous matrix in the following manner:

$$  \mathbf{A}^\text{T} \mathbf{A} = \begin{bmatrix}
	\vec{a}_1 \cdot \vec{a}_1& \vec{a}_1 \cdot  \vec{a}_2 & \vec{a}_1 \cdot \vec{a}_3  \\
	\vec{a}_2 \cdot \vec{a}_1& \vec{a}_2 \cdot  \vec{a}_2 & \vec{a}_2 \cdot \vec{a}_3  \\
	\vec{a}_3 \cdot \vec{a}_1& \vec{a}_3 \cdot  \vec{a}_2 & \vec{a}_3 \cdot \vec{a}_3
\end{bmatrix}  $$

If $\mathbf{A}$ is an orthogonal matrix then its transpose $\mathbf{A}^\text{T}$ should be equal to the inverse $\mathbf{A}^{-1}$. This means the above action would result in the identity matrix:

$$  \mathbf{A}^\text{T} \mathbf{A}  = \begin{bmatrix}
	\vec{a}_1 \cdot \vec{a}_1& \vec{a}_1 \cdot  \vec{a}_2 & \vec{a}_1 \cdot \vec{a}_3  \\
	\vec{a}_2 \cdot \vec{a}_1& \vec{a}_2 \cdot  \vec{a}_2 & \vec{a}_2 \cdot \vec{a}_3  \\
	\vec{a}_3 \cdot \vec{a}_1& \vec{a}_3 \cdot  \vec{a}_2 & \vec{a}_3 \cdot \vec{a}_3
\end{bmatrix} = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & 1 & 0 \\
	0 & 0 & 1
\end{bmatrix}  $$

For this to be true, it must be that the three vectors used to create the matrix have unit magnitude (equating the main diagonal elements):

$$ \vec{a}_1 \cdot \vec{a}_1 = 1 \qquad \vec{a}_2 \cdot \vec{a}_2 = 1 \qquad \vec{a}_3 \cdot \vec{a}_3 = 1 $$

and are mutually orthogonal (equating the off diagonal elements):

$$ \vec{a}_1 \cdot \vec{a}_2 = 0 \qquad \vec{a}_1 \cdot \vec{a}_3 = 0 \qquad \vec{a}_2 \cdot \vec{a}_3 = 0 $$

This is why we call matrices of this kind **orthogonal**, because they can be thought of as being constructed from vectors that are mutually orthogonal and have magnitudes of 1. 

You can do a similar proof with $\mathbf{A} \mathbf{A}^\text{T}$, though it is slightly less straightforward.


{% capture ex %}
Rotation matrices and reflection matrices are orthogonal. How can we check? Let take one of the rotation matrices given in the review, say:

$$ \mathbf{R}_x(\alpha) = \begin{bmatrix}
   1 & 0 & 0 \\
   0 & \cos(\alpha) & -\sin(\alpha) \\
   0 & \sin(\alpha) & \cos(\alpha) 
\end{bmatrix} $$ 

if we take its transpose, we get:

$$\mathbf{R}_x(\alpha)^\text{T} = \begin{bmatrix}
   1 & 0 & 0 \\
   0 & \cos(\alpha) & \sin(\alpha) \\
   0 & -\sin(\alpha) & \cos(\alpha) 
\end{bmatrix} $$

Now, let's act that on the original matrix. If the transpose of this matrix is truly its inverse, then we should get an identity matrix back:

$$
\begin{aligned}
   \mathbf{R}_x(\alpha)^\text{T} \mathbf{R}_x(\alpha) &= \begin{bmatrix}
      1 & 0 & 0 \\
      0 & \cos(\alpha) & \sin(\alpha) \\
      0 & -\sin(\alpha) & \cos(\alpha) 
   \end{bmatrix} \begin{bmatrix}
      1 & 0 & 0 \\
      0 & \cos(\alpha) & -\sin(\alpha) \\
      0 & \sin(\alpha) & \cos(\alpha) 
   \end{bmatrix}\\[1.0ex]
   &= \begin{bmatrix}
      1 & 0 & 0 \\
      0 & 1 & -\cos(\alpha)\sin(\alpha) + \sin(\alpha)\cos(\alpha) \\
      0 & -\sin(\alpha)\cos(\alpha) + \cos(\alpha)\sin(\alpha) & 1
   \end{bmatrix} \\[1.0ex]
   &= \begin{bmatrix}
      1 & 0 & 0 \\
      0 & 1 & 0 \\
      0 & 0 & 1
   \end{bmatrix} \\[1.0ex]
   \mathbf{R}_x(\alpha)^\text{T} \mathbf{R}_x(\alpha) &= \mathbf{I}
\end{aligned}
$$

where we made use of the fact that $\cos^2(\alpha) + \sin^2(\alpha) = 1$ to simplify going from the first to second line.

You can show that acting in the opposite order gives the same result. This shows that the transpose of this rotation matrix is, in fact, its inverse. This is true for all rotation and reflection matrices. 
{% endcapture %}
{% include example.html content=ex %}
















## Application:


Consider a rigid body rotating in three-dimensional space. The rotation is described by a rotation matrix $ \mathbf{R}(\theta) $, the angular velocity vector is $ \vec{\omega} $, and the moment of inertia tensor (a matrix) is $ \mathbf{I} $, which is symmetric. We want to calculate the angular momentum $ \vec{L} $ of the rigid body.

The angular momentum $ \vec{L} $ is given by the formula:
$$
\vec{L} = \mathbf{I} \vec{\omega}
$$


**Define the Matrices and Vectors**

Let the moment of inertia tensor for a body be represented as:
$$
\mathbf{I} = \begin{pmatrix} 
	I_{xx} & 0 & 0 \\
	0 & I_{yy} & 0 \\
	0 & 0 & I_{zz} 
\end{pmatrix}
$$
where we will suppose $ I_{xx} = 2 \, \text{kg} \cdot \text{m}^2 $, $ I_{yy} = 3 \, \text{kg} \cdot \text{m}^2 $, and $ I_{zz} = 4 \, \text{kg} \cdot \text{m}^2 $.

Let the angular velocity vector be:
$$
\vec{\omega} = \begin{pmatrix} 
	\omega_x \\
	\omega_y \\
	\omega_z 
\end{pmatrix} = \begin{pmatrix} 
	1 \, \text{rad/s} \\
	2 \, \text{rad/s} \\
	3 \, \text{rad/s} 
\end{pmatrix}
$$

**Calculate the Angular Momentum**

The angular momentum $ \vec{L} $ is given by:

$$
\vec{L} = \mathbf{I} \vec{\omega} = \begin{pmatrix} 
	I_{xx} & 0 & 0 \\
	0 & I_{yy} & 0 \\
	0 & 0 & I_{zz} 
\end{pmatrix} \begin{pmatrix} 
	\omega_x \\
	\omega_y \\
	\omega_z 
\end{pmatrix}
= \begin{pmatrix} 
	I_{xx} \omega_x \\
	I_{yy} \omega_y \\
	I_{zz} \omega_z 
\end{pmatrix}
$$

Substituting the values to get the angular momentum gives:

$$
\vec{L} = \begin{pmatrix} 
	2 \cdot 1 \\
	3 \cdot 2 \\
	4 \cdot 3 
\end{pmatrix}
= \begin{pmatrix} 
	2 \, \text{kg} \cdot \text{m}^2/\text{s} \\
	6 \, \text{kg} \cdot \text{m}^2/\text{s} \\
	12 \, \text{kg} \cdot \text{m}^2/\text{s} 
\end{pmatrix}
$$

**Rotation Matrix (Orthogonal Matrix)**

Suppose the body undergoes a rotation described by a rotation matrix $ \mathbf{R}(\theta) $. Let the rotation matrix be a simple 90-degree rotation about the $ z $-axis given by:

$$
\mathbf{R}_z(90^\circ) = \begin{pmatrix}
	\cos 90^\circ & -\sin 90^\circ & 0 \\
	\sin 90^\circ & \cos 90^\circ & 0 \\
	0 & 0 & 1
\end{pmatrix}
= \begin{pmatrix}
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{pmatrix}
$$

Applying the rotation to the angular momentum, we get:

$$
\vec{L'} = \mathbf{R}_z(90^\circ) \vec{L} = \begin{pmatrix}
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{pmatrix} \begin{pmatrix}
	2 \\
	6 \\
	12
\end{pmatrix}
= \begin{pmatrix}
	-6 \, \text{kg} \cdot \text{m}^2/\text{s} \\
	2 \, \text{kg} \cdot \text{m}^2/\text{s} \\
	12 \, \text{kg} \cdot \text{m}^2/\text{s}
\end{pmatrix}
$$

**Symmetric and Antisymmetric Matrices**

Now, consider the cross product $ \vec{\omega} \times \vec{L} $, which determines the torque responsible for the precession of rigid bodies. For example, the angular momentum and angular velocity vectors of the Earth are not perfectly aligned. This misalignment generates a precessional torque, causing the Earth's axis to slowly trace out a circular motion, which results in the gradual shift of the ``North'' star over time.
  

The torque responsible for the precession cross product can be expressed using the cross matrix of the angular velocity. In this case, the matrix $ [\vec{\omega}]_\times $, represents the cross product operation:

$$
[\vec{\omega}]_\times = \begin{pmatrix}
	0 & -\omega_z & \omega_y \\
	\omega_z & 0 & -\omega_x \\
	-\omega_y & \omega_x & 0
\end{pmatrix}
= \begin{pmatrix}
	0 & -3 & 2 \\
	3 & 0 & -1 \\
	-2 & 1 & 0
\end{pmatrix}
$$

We can now compute $ \vec{\omega} \times \vec{L} $ using matrix multiplication:

$$
\begin{aligned}
\vec{\omega} \times \vec{L} &= [\vec{\omega}]_\times \vec{L}\\
&= \begin{pmatrix}
	0 & -3 & 2 \\
	3 & 0 & -1 \\
	-2 & 1 & 0
\end{pmatrix} \begin{pmatrix}
	2 \\
	6 \\
	12
\end{pmatrix}\\
&= \begin{pmatrix}
	0\cdot2 + (-3)\cdot6 + 2\cdot12 \\
	3\cdot2 + 0\cdot6 + (-1)\cdot12 \\
	(-2)\cdot2 + 1\cdot6 + 0\cdot12
\end{pmatrix}\\
&= \begin{pmatrix}
	12 - 18 + 24 \\
	6 - 12 \\
	-4 + 6
\end{pmatrix}\\
&= \begin{pmatrix}
	18 \\
	-6 \\
	2
\end{pmatrix}
\end{aligned}
$$
 

 










## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


Consider a 3-dimensional rigid body with a moment of inertia tensor (a matrix) $ \mathbf{I} $, which is used to describe the relationship between the angular momentum $ \vec{L} $ and the angular velocity $ \vec{\omega} $ through the equation:

$$
\vec{L} = \mathbf{I} \vec{\omega}
$$

The following matrix representations are provided:


- The **identity matrix** $ \mathbf{I}_3 $, which represents the isotropic moment of inertia:

$$
\mathbf{I}_3 = \begin{bmatrix}
   1 & 0 & 0 \\
   0 & 1 & 0 \\
   0 & 0 & 1
\end{bmatrix}
$$
	
- A **diagonal matrix** $ \mathbf{I}_\text{diag} $, representing a symmetric rigid body:

$$
\mathbf{I}_\text{diag} = \begin{bmatrix}
   3 & 0 & 0 \\
   0 & 2 & 0 \\
   0 & 0 & 1
\end{bmatrix} \, \mathrm{kg \cdot m^2}
$$

- A **symmetric matrix** $ \mathbf{I}_\text{sym} $, which represents a rigid body where the moments of inertia along different axes are not independent:

$$
\mathbf{I}_\text{sym} = \begin{bmatrix}
   4 & 1 & 0 \\
   1 & 3 & 0 \\
   0 & 0 & 2
\end{bmatrix} \, \mathrm{kg \cdot m^2}
$$

- An **antisymmetric matrix** $ \mathbf{A} $, which represents the cross-product operation for a vector $ \vec{\omega} = \begin{bmatrix} \omega_x & \omega_y & \omega_z \end{bmatrix}^\text{T} $:

$$
\boldsymbol{\omega}_{\times}  = \begin{bmatrix}
   0 & -\omega_z & \omega_y \\
   \omega_z & 0 & -\omega_x \\
   -\omega_y & \omega_x & 0
\end{bmatrix}
$$

- An **orthogonal matrix** $ \mathbf{R} $, representing a rotation about the $ z $-axis by an angle $ \theta = \frac{\pi}{4} $:

$$
\mathbf{R}_z(\tfrac{\pi}{4}) = \begin{bmatrix}
   \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} & 0 \\
   \tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} & 0 \\
   0 & 0 & 1
\end{bmatrix}
$$



a) Verify that the angular momentum $ \vec{L} $ is the same (ignoring the units) as the angular velocity vector when the moment of inertia is represented by the identity matrix $ \mathbf{I}_3 $, and 

$$ \vec{\omega} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} $$

b) Compute $ \vec{L} $ when the moment of inertia is $ \mathbf{I}_\text{diag} $ and the angular velocity is the same as the one given in part (a).

c) Show that $ \mathbf{I}_\text{sym} $ is symmetric by explicitly verifying $ \mathbf{I}_\text{sym} = \mathbf{I}_\text{sym}^\text{T} $. Then compute $ \vec{L} $ using the same $\vec{\omega}$ as part (a).

d) Write the cross product matrix for $\vec{\omega}$ to calculate torque vector $ \vec{\tau} = \vec{\omega} \times  \vec{L} $, where

$$ \vec{L} = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix} \qquad \qquad \vec{\omega} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} $$

e) Rotate the angular momentum vector 

$$ \vec{L} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$

using the orthogonal matrix $\mathbf{R}_z(\tfrac{\pi}{4})  $ given in the problem statement. Also, verify that $ \mathbf{R}_z(\tfrac{\pi}{4}) ^\text{T} \mathbf{R}_z(\tfrac{\pi}{4})  = \mathbf{I}_3 $, confirming the orthogonality of the rotation matrix.
