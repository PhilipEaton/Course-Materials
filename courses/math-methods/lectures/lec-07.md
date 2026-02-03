---
layout: default
title: Mathematical Methods - Lecture 07
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 7
---

# Lecture 07 – Complex Numbers, Matrices, and a New Representation
	


## Complex Numbers as 2-Dimensional Coordinates

Complex numbers naturally show up when solving equations like $x^2 + 1 = 0$. By defining the imaginary unit $i$ as $i = \sqrt{-1}$, we can write any complex number $z$ in the form:

$$
z = a + bi
$$

where $a$ and $b$ are real numbers. With this simple definition, we unlock an entire world of mathematics called **complex algebra**.

To be begin this discussion, it is particularly convenient to think of complex numbers as points in a Cartesian-like coordinate system, where the horizontal axis represents the real part of $ z $ (tha is, $ a $) and the vertical axis represents the imaginary part ($ b $). This geometric interpretation provides an intuitive way to visualize and work with complex numbers.

<div class="two-column">

<div class="column">

<img
  src="{{ '/courses/math-methods/images/lec07/MMLec07Fig1.png' | relative_url }}"
  alt="The image shows a standard coordinate plane with a horizontal axis labeled “Real” and a vertical axis labeled “Imag.” A point representing the complex number is plotted in the first quadrant. From this point, a dashed vertical line drops straight down to meet the real axis at a positive value labeled “a,” and a dashed horizontal line extends leftward to meet the imaginary axis at a positive value labeled “b.” The point itself is labeled “a plus b i.” The axes are drawn with arrows indicating positive direction."
  style="display:block; margin:1.5rem auto; max-width:400px; width:80%;">

</div>

<div class="column">

<img
  src="{{ '/courses/math-methods/images/lec07/MMLec07Fig2.png' | relative_url }}"
  alt="The image shows the same coordinate plane with a horizontal axis labeled “Real” and a vertical axis labeled “Imag.” A point representing the same complex number lies in the first quadrant. A dashed line connects the origin to the point, representing the radius in polar form. An angle at the origin, measured from the positive real axis up to the dashed line, indicates the direction of the point. This angle is labeled “theta,” and the dashed radius line is labeled “r.” The point itself is labeled “a plus b i.”"
  style="display:block; margin:1.5rem auto; max-width:400px; width:80%;">

</div>

</div>


Looking at the diagrams, we can describe the complex number $z$ using **polar coordinates**, where:

$$
r = \sqrt{a^2 + b^2} \qquad\text{and}\qquad \theta = \tan^{-1}\left(\frac{b}{a}\right)
$$

Here, $r$ is called the **magnitude** or **modulus** of $z$, and $\theta$ is called the **phase** or **argument**.

We can relate the Cartesian form to the polar form using a bit of trigonometry:

$$
a = r \cos(\theta) \qquad \text{and} \qquad b = r \sin(\theta)
$$

Plugging this into the original expression for $z$ gives:

$$
z = a + i b = r \cos(\theta) + i r \sin(\theta) = r \left( \cos(\theta) + i \sin(\theta) \right)
$$

This formulation will become very important in a bit.





## Complex Numbers as Rotations

Earlier, we saw how the **dot product** of two vectors, say $\vec{v}$ and $\vec{u}$, can be expressed in terms of the angle between them:

$$ 
\vec{v} \cdot \vec{u} = v u \cos(\theta_{vu})
$$

where $\theta_{vu}$ is the smallest angle between the vectors. Rearranging this allows us to solve for the angle between the two vectors:

$$ 
\cos(\theta_{vu}) = \frac{\vec{v} \cdot \vec{u}}{v u}
$$

So, if we know how to write vectors in terms of basis directions (like $\hat{i}$, $\hat{j}$, or $\hat{k}$), we can find the angle between them using just the dot product.

Let’s dream a little: **What if we treat complex numbers like vectors?**

We could take $\widehat{\text{real}}$ to be the units vector pointing along the real-axis, and similalry $\widehat{\text{imag}}$ to point along the imaginary-axis. So, a complex number could be written as a vector in the following manner:

$$
\vec{z} = a\ \ \widehat{\text{real}} \ \ + \ \  b\ \ \widehat{\text{imag}}
$$

This is just a new way to represent $z = a + ib$. And once we’ve written it as a vector, we can use vector tools to analyze it.

For example, to find the angle between $\vec{z}$ and the real axis, we can compute:

$$
\vec{z} \cdot \widehat{\text{real}} = \vert\vec{z}\vert \cdot \vert\widehat{\text{real}}\vert \cdot \cos(\theta)
$$

Since $\vec{z} \cdot \widehat{\text{real}}$ is just the real component $a$, $\vert\widehat{\text{real}}\vert = 1$, and $\vert\vec{z}\vert = r$, this simplifies to:

$$
a = r \cos(\theta) \quad \implies \quad \cos(\theta) = \frac{a}{r}
$$

Which matches exactly what we’d expect from polar coordinates. This suggests our vector idea isn't totally insain!


Let’s push this further. In regular algebra, we can multiply two complex numbers together. So let’s try multiplying:

- $z = a + bi$
- $w = \cos(\phi) + i \sin(\phi)$
	- A complex number of a magnitude of 1 set at and angle of $\phi$ from the real axis.
	- Compare with how we wrote $z$ above to see for yourself.

We’ll call the result $z' = wz$. Let’s compute it using basic albegra:

$$
\begin{aligned}
	z' &= \big(\cos(\phi) + i \sin(\phi)\big) \cdot \big(a + bi\big) \\
	   &= a \cos(\phi) + a i \sin(\phi) + b i \cos(\phi) - b \sin(\phi) \\
	   &= \big(a \cos(\phi) - b \sin(\phi)\big) + i\big(a \sin(\phi) + b \cos(\phi)\big)
\end{aligned}
$$

This currently looks like a mess, but we promise it reveals something important. Let's find the angle that $z'$ makes with the real axis. First, we’ll write $z'$ as a vector:

$$
\vec{z}' = (a \cos(\phi) - b \sin(\phi))\ \widehat{\text{real}} \ \ +\ \  (a \sin(\phi) + b \cos(\phi))\ \widehat{\text{imag}}
$$

To find the new angle $\theta'$, we again use:

$$
\cos(\theta') = \frac{\text{real part of } z'}{\vert\vec{z}'\vert} = \frac{a \cos(\phi) - b \sin(\phi)}{\vert\vec{z}'\vert}
$$

Looks like we need the magnitude of $\vec{z}'$:

$$
\begin{aligned}
	\vert \vec{z}' \vert^2 &= (a \cos(\phi) - b \sin(\phi))^2 + (a \sin(\phi) + b \cos(\phi))^2 \\[1.15ex]
	&= a^2 \cos^2(\phi) - 2ab \cos(\phi)\sin(\phi) + b^2 \sin^2(\phi) \ \ + \\
	&\ \ \ \ \ \ + a^2 \sin^2(\phi) + 2ab \sin(\phi)\cos(\phi) + b^2 \cos^2(\phi) \\[1.15ex]
	&= a^2 (\cos^2(\phi) + \sin^2(\phi)) + b^2 (\cos^2(\phi) + \sin^2(\phi)) \\[1.15ex]
	&= a^2 + b^2 \\[1.15ex]
	\vert \vec{z}' \vert^2  &= r^2
\end{aligned}
$$

So, $\vert\vec{z}'\vert = r$, just like the original complex number.

Now we can go back to the angle off of the real axis we were calculating for $\vec{z}'$:

$$
\begin{aligned}
\cos(\theta') &= \frac{a \cos(\phi) - b \sin(\phi)}{r} \\
              &= \frac{a}{r} \cos(\phi) - \frac{b}{r} \sin(\phi) \\
              &= \cos(\theta) \cos(\phi) - \sin(\theta) \sin(\phi) \\
\cos(\theta') &= \cos(\theta + \phi)
\end{aligned}
$$

where we have used an angle addition identity at the end to simplify into the final expression.

Let's take stock of these results. We multiplied the complex number $z$ by another complex number $w$ to get $z' = w z$, and found:

- The magnitude didn’t change: $|z'| = |z|$
- The angle increased by $\phi$: $\theta' = \theta + \phi$

In other words, **multiplying by $w$ appears to have rotated $z$ by an angle $\phi$**. That’s pretty wild!

Complex multiplication doesn’t just scale numbers, it can rotate them!

{% capture ex %}
Multiplying a complex number $ z $ by the complex number:  

$$
w = \cos(\phi) + \sin(\phi) \, i  
$$  

results in a rotation of $ z $ by an angle $ \phi $ in the counterclockwise direction.  
{% endcapture %}
{% include result.html content=ex %}
	








## Complex Numbers as Matrices 

### Discovery

Recall, we know how to write rotation of a 2-dimensional plane  in terms of a $2\times 2$ rotation matrix. So, let's perform a little change of representation and instread of writing the complex number $ w $ as a vector, let's write it as a rotation matrix.   

From before, we found that multiplying a complex number by: 

$$w = \cos(\phi) + i \sin(\phi)$$

rotates it by an angle $\phi$. In vector form, this process can written as:

$$
z' = w z \quad \Longrightarrow \quad \vec{z}' = \mathbf{R}(\phi)\ \vec{z}
$$

The corresponding rotation matrix is:

$$
\mathbf{R}(\phi) = 
\begin{bmatrix}
\cos(\phi) & -\sin(\phi) \\
\sin(\phi) & \cos(\phi)
\end{bmatrix}
$$

We would like this matrix to resemble the complex number $w = \cos(\phi) + i \sin(\phi)$ more closely. To do this, we can separate the rotation matrix into its cosine and sine parts:

$$
\begin{aligned}
\mathbf{R}(\phi) &= \begin{bmatrix}  
	\cos(\phi) & 0 \\  
	0 & \cos(\phi)  
\end{bmatrix} + \begin{bmatrix}  
	0 & -\sin(\phi) \\  
	\sin(\phi) & 0  
\end{bmatrix} \\[4ex]
\mathbf{R}(\phi) &= \cos(\phi) \begin{bmatrix}  
	1 & 0 \\  
	0 & 1  
\end{bmatrix} + \sin(\phi)\begin{bmatrix}  
	0 & -1 \\  
	1 & 0  
\end{bmatrix}  
\end{aligned}
$$  

Compare this with the complex number

$$
w = \cos(\phi) + \sin(\phi)\, i
= \cos(\phi)\cdot 1 + \sin(\phi)\cdot i.
$$

This comparison suggests the following identifications:

- The number $1$ corresponds to the identity matrix:
  
  $$
  1 \;\longrightarrow\;
  \begin{bmatrix}
  1 & 0 \\
  0 & 1
  \end{bmatrix}
  $$

- The imaginary unit $i$ corresponds to the matrix

  $$
  i \;\longrightarrow\;
  \begin{bmatrix}
  0 & -1 \\
  1 & 0
  \end{bmatrix}
  $$

With these rerepresentations, the complex number $w$ can be written directly as a matrix:

$$
\mathbf{w} = \begin{bmatrix}
\cos(\phi) & -\sin(\phi) \\
\sin(\phi) & \cos(\phi)
\end{bmatrix}
$$

This agrees exactly with what we found earlier when we multiplied $z$ by $w$. In matrix form, $w$ acts as a rotation operator on the vector representation of the complex number $z$. Great!











### Testing

This re-representation of $1$ and $i$ in matrix form, assuming it is correct, allows us to write *any* complex number as a matrix instead of as a scalar or vector. For example, we can rewrite $z = a + b i$ as:

$$ 
\mathbf{z} = a \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} + b \begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix} = \begin{bmatrix}
a & -b \\
b & a
\end{bmatrix} 
$$

This is the complex number $z$ written in matrix form, or so we claim.

It is worth connecting this representation with the matrix operations we already know and seeing what they mean in terms of complex numbers. If the results make sense, then this representation is doing something reasonable. If the results are strange or contradictory, then we would need to rethink our assumptions.

- **Determinant**: The determinant of $\mathbf{z}$ is calculated to be:

	$$ \text{det}(\mathbf{z}) = \begin{vmatrix}
	a & -b \\
	b & a
	\end{vmatrix} = a^2 - (-b^2) = a^2 + b^2 = r^2 $$

	The determinant of $\mathbf{z}$ appears to represent the magnitude-squared of the complex number. 


- **Transpose**: The transpose of $\mathbf{z}$:

	$$ \mathbf{z}^\text{T} = \begin{bmatrix}
	a & b \\
	-b & a
	\end{bmatrix}$$
	
	To discover what this represents, let's rewrite this in terms of the identity matrix $\mathbf{I}$ and the matrix representation of the inaginary unit $\mathbf{i}$. This gives:  
	
	$$\mathbf{z}^\text{T} = a \begin{bmatrix}
	1 & 0 \\
	0 & 1
	\end{bmatrix} - b \begin{bmatrix}
	0 & -1 \\
	1 & 0
	\end{bmatrix} = a\mathbf{I} - b\mathbf{i} $$

	So, the transpose appears to represent the complex conjugation (take $i \rightarrow -i$) of the complex number.


- **Inverse**: The inverse of $\mathbf{z}$, found using the equation for the inverse of s $2\times 2$ matrix:

	$$ \mathbf{z}^{-1} = \frac{1}{\det(\mathbf{z})} \begin{bmatrix}
	a & b \\
	-b & a
	\end{bmatrix} = \frac{1}{r^2}\  \mathbf{z}^\text{T} = \frac{\mathbf{z}^*}{r^2}   $$

	This one is hard to see, but consider the simplification of $1/z$ by removing all instances of $i$ from the denominator:

	$$ z^{-1} =  \frac{1}{z} = \frac{1}{a + i b} = \frac{1}{a + i b} \left(\frac{a - i b}{a - i b}\right) = \frac{z^*}{a^2 + b^2} = \frac{z^*}{r^2}  $$ 

	The inverse of the matrix representation is essentially identical the taking the inverse of (1 over...) the complex number.


- **Symmetric and Antisymmetric Decomposition**:
	- The symmetric component of $\mathbf{z}$ can be found as:

	$$ \mathbf{z}_S = \frac{1}{2} \left( \mathbf{z} + \mathbf{z}^\text{T}  \right) = \frac{1}{2} \left( \begin{bmatrix}
		a & -b \\
		b & a
	\end{bmatrix} + \begin{bmatrix}
		a & b \\
		-b & a
	\end{bmatrix}  \right) = \begin{bmatrix}
		a & 0 \\
		0 & a
	\end{bmatrix} = a \mathbf{I}  $$

	is the real part of the complex number $z$. 

	- The antisymmetric component of $\mathbf{z}$ an be found as:

	$$ \mathbf{z}_A = \frac{1}{2} \left( \mathbf{z} - \mathbf{z}^\text{T}  \right) = \frac{1}{2} \left( \begin{bmatrix}
		a & -b \\
		b & a
	\end{bmatrix} - \begin{bmatrix}
		a & b \\
		-b & a
	\end{bmatrix}  \right) = \begin{bmatrix}
		0 & -b \\
		b & 0
	\end{bmatrix} = b \mathbf{i} $$

	is the imaginary part of the complex number $z$.

	Writing $\mathbf{z}$ in terms of its symmetric and antisymmetric components is identical to writing the complex number in terms of its real and imaginary parts. 


It is striking that the matrix operations we have studied so far all have natural interpretations when complex numbers are written as matrices. This is strong evidence that we are still doing ordinary algebra, just expressed in a different mathematical language.












## Complex Matrices

A **complex matrix** is a matrix whose elements can be complex numbers. These matrices generalize many properties of real matrices while enabling operations and transformations in spaces where both magnitude and phase (angle) are important, as we will see.

In the way manner as real matrices, complex matrices allow for addition, subtraction, scalar multiplication, and matrix multiplication, where the underlying arithmetic is governed by the rules of complex numbers.

A key difference for complex matrices is the availability of additional operations. For example, you can compute the **complex conjugate of the elements** of the matrix:

$$
\mathbf{A} = \begin{bmatrix}
	a_{11} & a_{12}\\
	a_{21} & a_{22}
\end{bmatrix} \qquad\implies\qquad \mathbf{A}^* = \begin{bmatrix}
	a_{11}^* & a_{12}^*\\
	a_{21}^* & a_{22}^*
\end{bmatrix}
$$

where $ \mathbf{A} $ is a complex matrix and $ \mathbf{A}^* $ is its complex conjugate. 


However, we can compute the transpose of a complex matrix:

$$
\mathbf{A} = \begin{bmatrix}
	a_{11} & a_{12}\\
	a_{21} & a_{22}
\end{bmatrix} \qquad \mathbf{A}^\text{T} = \begin{bmatrix}
	a_{11} & a_{21}\\
	a_{12} & a_{22}
\end{bmatrix}
$$

where we recall that if the matrix represents a complex number, then the transpose of the matrix is the complex conjugate of the complex number the matrix is representing. 

This opens us to a new, and very powerful matrix operation. Combining the complex conjugation of the elements and the transpose (representing the complex conjugation of the matrix), we get an operation known as the **Hermitian adjoint** (or simply the **adjoint**):

$$
\mathbf{A}^\dagger = \begin{bmatrix}
	a_{11}^* & a_{21}^*\\
	a_{12}^* & a_{22}^*
\end{bmatrix}
$$

where $\dagger$ is read "dagger" representing the Hermitian adjoint of the matrix. This can be thought of as the *full complex conjugation* of a matrix since you are conjugating the matrix itself and its individual elements. 


The adjoint leads to some vital classifications of complex matrices:
- **Hermitian Matrices**: 
	- Matrices that satisfy $ \mathbf{A} = \mathbf{A}^\dagger $. 
	- These are the complex analogs of symmetric matrices.
- **Unitary Matrices**: 
	- Matrices that satisfy $ \mathbf{U}^\dagger \mathbf{U} = \mathbf{I} $, representing length-preserving transformations (such as rotations in complex space).
	- These are the complex analogs of orthogonal matrices.
- **Normal Matrices**: 
	- Matrices that satisfy $ \mathbf{A}^\dagger \mathbf{A} = \mathbf{A} \mathbf{A}^\dagger $, a condition that generalizes diagonalizability, as we will see later. 
	- Since a Hermitian matrix is equal to its own Hermitian adjoint ($ \mathbf{A} = \mathbf{A}^\dagger $), the Normal matrix condition ($ \mathbf{A}^\dagger \mathbf{A} = \mathbf{A} \mathbf{A}^\dagger $) will be automatically satisfied. Thus, all Hermitian matrices are also Normal, but not all Normal matrices are Hermitian. In other words, Hermitian matrices form a subset of the broader category of Normal matrices.


{% capture ex %}
Suppose we are given the following matrices:

$$
\mathbf{A} = \begin{bmatrix}
	2 & i \\
	-i & 3
\end{bmatrix} \qquad \qquad
\mathbf{B} = \frac{1}{\sqrt{2}} \begin{bmatrix}
	1 & i \\
	-i & 1
\end{bmatrix} \qquad \qquad
\mathbf{C} = \begin{bmatrix}
	0 & 1 + i \\
	1 - i & 0
\end{bmatrix}
$$

and we are tasked with check to see if they are Hermitian, Unitary, and/or Normal matrices. Let's check by check if each matrix has the property that defined each classification.

**Checking Hermitian Property ($ \mathbf{A} = \mathbf{A}^\dagger $)**

- **Matrix $\mathbf{A}$**:  
	The adjoint (conjugate then transpose) is:

	$$
	\mathbf{A}^\dagger = \left(\begin{bmatrix}
		2 & i \\
		-i & 3
	\end{bmatrix}^{\text{T}}\right)^{*} = \left(\begin{bmatrix}
	2 & -i \\
	i & 3
	\end{bmatrix}\right)^{*} = \begin{bmatrix}
	2 & i \\
	-i & 3
	\end{bmatrix} = \mathbf{A}
	$$

	Since $ \mathbf{A}^\dagger = \mathbf{A} $, $\mathbf{A}$ is **Hermitian**.

- **Matrix $\mathbf{B}$**:  
	The adjoint is:

	$$
	\mathbf{B}^\dagger = \frac{1}{\sqrt{2}} \left(\begin{bmatrix}
		1 & i \\
		-i & 1
	\end{bmatrix}^\text{T} \right)^{*} = \frac{1}{\sqrt{2}} \left(\begin{bmatrix}
	1 & -i \\
	i & 1
	\end{bmatrix} \right)^{*} = \frac{1}{\sqrt{2}} \begin{bmatrix}
	1 & i \\
	-i & 1
	\end{bmatrix} = \mathbf{B}
	$$

	Since $ \mathbf{B}^\dagger = \mathbf{B} $, $\mathbf{B}$ is **Hermitian**.

- **Matrix $\mathbf{C}$**:  
	The adjoint is:

	$$
	\mathbf{C}^\dagger = \left(\begin{bmatrix}
		0 & 1 + i \\
		1 - i & 0
	\end{bmatrix}^\text{T} \right)^* = \left(\begin{bmatrix}
	0 & 1 - i \\
	1 + i & 0
	\end{bmatrix} \right)^* = \begin{bmatrix}
	0 & 1 + i \\
	1 - i & 0
	\end{bmatrix}  = \mathbf{C} 
	$$

	Since $ \mathbf{C}^\dagger = \mathbf{C} $, $\mathbf{C}$ is **Hermitian**.


**Checking Unitary Property ($ \mathbf{B}^\dagger \mathbf{B} = \mathbf{I} $)**

- **Matrix $\mathbf{A}$**:  
	Checking this property directly:

	$$
	\mathbf{A}^\dagger \mathbf{A} = \begin{bmatrix}
		2 & i \\
		-i & 3
	\end{bmatrix} \begin{bmatrix}
	2 & i \\
	-i & 3
	\end{bmatrix} = \begin{bmatrix}
	5 & 5i \\
	-5i & 10
	\end{bmatrix} \ne \mathbf{I}
	$$

	So, $\mathbf{A}$ is **not Unitary**.


- **Matrix $\mathbf{B}$**:  
	Checking this property directly:

	$$
	\mathbf{B}^\dagger \mathbf{B} = \frac{1}{2} \begin{bmatrix}
		1 & i \\
		-i & 1
	\end{bmatrix} \begin{bmatrix}
	1 & i \\
	-i & 1
	\end{bmatrix} = \frac{1}{2} \begin{bmatrix}
		2 & 2i \\
		-2i & 2
	\end{bmatrix} \ne \mathbf{I}
	$$

	So, $\mathbf{B}$ is **not Unitary**.

- **Matrix $\mathbf{C}$**:  
	Checking this property directly:

	$$
	\mathbf{C}^\dagger \mathbf{C} = \begin{bmatrix}
		0 & 1 + i \\
		1 - i & 0
	\end{bmatrix} \begin{bmatrix}
	0 & 1 + i \\
	1 - i & 0
	\end{bmatrix} = \begin{bmatrix}
		2 & 0 \\
		0 & 2
	\end{bmatrix} = 2 \mathbf{I} \ne \mathbf{I}
	$$

	So, $\mathbf{C}$ is **not Unitary**.


**Checking Normal Property ($ \mathbf{A}^\dagger \mathbf{A} = \mathbf{A} \mathbf{A}^\dagger $)**
Since all of these are Hermitian, all of these matrices will be Normal we well. 

{% endcapture %}
{% include example.html content=ex %}
	











## Quaternions

When thinking about complex numbers and their representation in 2-dimensional space, one might begin to wonder what other dimensional spaces can be expressed using complex, or complex-like, numbers. In 1842 William Rowan Hamilton stumbled upon an idea that opened complex-like numbers up to a 4-dimensional space (1 real and 3 imaginary). Numbers written in this space are called quaternions and are used extensively throughout physics, especially in quantum mechanics and quantum electrodynamics.

A quaternion is written mathematically in the following manner: 

$$
q = a + b i + c j + d k
$$

where $a, b, c,$ and $d$ are real numbers, and the imaginary units $i, j, k$ satisfy the following rules:

$$
i^2 = j^2 = k^2 = ijk = -1
$$

and

$$
ij = k, \quad ji = -k, \quad jk = i, \quad kj = -i, \quad ki = j, \quad ik = -j
$$

the order matters!

Notice that these relations resemble the cross product, and the letters used ($i, j, k$) are the same as the ones used for the unit vectors for the $x$-, $y$-, and $z$-axes. This is not a coincidence. Quaternions were developed before vectors and their operations were formally introduced. In fact, vectors and their operations emerged as an attempt to simplify the complicated mathematics involved in quaternion manipulations. This development was done in the mid-1880's by Oliver Heaviside, Hermann von Helmholtz, and Josiah Willard Gibbs, some names you may recognize from other physics courses. 

Similar to how complex numbers could be represented as a couple of $2\times 2$ real matrices, quaternions can be represented as four $2 \times 2$ complex matrices. You can begin by assuming you have four arbitrary $2\times 2$ matrices that obey the relations between $i, j, k$ indicated above. Working through all of those relations and demanding that each resulting matrix have a determinant of 1, you can find what each of the matrix needs to be. Without having to go through all of that busy work, we can write the basis elements of quaternions as:

$$
1 \rightarrow \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \quad
i \rightarrow \begin{bmatrix} i & 0 \\ 0 & -i \end{bmatrix} \quad
j \rightarrow \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} \quad
k \rightarrow \begin{bmatrix} 0 & i \\ i & 0 \end{bmatrix}
$$

These matrices are all *special* (i.e., they have a determinant of 1) and are *unitary* ($\mathbf{U}^\dagger = \mathbf{U}^{-1}$), you can check this on your own. It turns out that these four matrices together form the complete Special Unitary Group in 2 dimensions, $ SU(2) $. This group is the ideal way to represent spin-$\tfrac{1}{2}$ particles in quantum mechanics, and is related to the Pauli Spin Matrices, which you will encounter in your undergraduate or graduate Quantum Mechanics course.


From these four matrices, a quaternion can be written as a single $2\times 2$ matrix in the following manner:

$$q = a + b i + c j + d k = \begin{bmatrix}
	a + i b & c + i d \\
	-c + i d & a - i b
\end{bmatrix} $$

We could also have decided to represent these matrices using $4\times 4$ matrices in a full 4-dimensional space. This is slightly more difficult to get, but the following matrices are what come about after forcing the relations above and demanding a determinant of 1:

$$
1 \rightarrow \begin{bmatrix}
	1 & 0 & 0 & 0 \\
	0 & 1 & 0 & 0 \\
	0 & 0 & 1 & 0 \\
	0 & 0 & 0 & 1
\end{bmatrix} \qquad
i \rightarrow \begin{bmatrix}
	0 & 1 & 0 & 0 \\
	-1 & 0 & 0 & 0 \\
	0 & 0 & 0 & 1 \\
	0 & 0 & -1 & 0
\end{bmatrix}$$

$$
j \rightarrow \begin{bmatrix}
	0 & 0 & 1 & 0 \\
	0 & 0 & 0 & -1 \\
	-1 & 0 & 0 & 0 \\
	0 & 1 & 0 & 0
\end{bmatrix} \qquad
k \rightarrow \begin{bmatrix}
	0 & 0 & 0 & 1 \\
	0 & 0 & 1 & 0 \\
	0 & -1 & 0 & 0 \\
	-1 & 0 & 0 & 0
\end{bmatrix}
$$

This representation is useful when working in quantum field theories. 

### Applications

Quaternions and their matrix representations, particularly through the $SU(2)$ group, have a wide range of applications in both physics. As we said, cand cannot stress enough, the $SU(2)$ group plays a crucial role in the description of spin-$\tfrac{1}{2}$ particles. The Pauli matrices, which are described by the matrix elements of $SU(2)$, provide a matrix representation for spin operators, allowing for the mathematical treatment of spin in quantum systems. This is particularly useful for understanding quantum states, spin precession, and the behavior of particles in magnetic fields.

In addition, $SU(2)$ is fundamental in the theory of angular momentum, where it helps describe rotations in quantum systems. It serves as the mathematical foundation for spin-orbit coupling in atomic physics and is deeply involved in the study of quantum entanglement and quantum computing, where operations on qubits are often represented using $SU(2)$ matrices.

Outside of quantum mechanics, quaternion representations also find applications in computer graphics, where they are used to represent 3D rotations. Unlike traditional matrix methods, quaternions avoid some of the numerical instability and other issues, providing more efficient and stable ways to interpolate rotations (e.g., in animations or robotics). Moreover, quaternions are employed in various fields like control theory, where their properties are leveraged in algorithms for rotation-based computations, and in the description of certain types of wave phenomena.
















## Application:

### Electronics

One place you see complex numbers in action is in when working with AC (Alternating Current) circuits. For example, in an AC circuit, the current passing through the circuit can be given as a complex number in the following manner:

$$ I(t) = I_0 w(t) =  I_0 (\cos(\omega t) + \sin(\omega t) \, i) =  I_0 \cos(\omega t) + I_0 \sin(\omega t) \, i $$

where $w(t)$ is the complex number the resulted in a rotation by $\omega t$ we saw previously (where the angle was written as $\phi$ instead of $\omega t$).

Now, you may panic and say, ``But current is not a complex thing! How is this allowed?" and you would be right to complain. To clear this up, it is generally an unsaid rule when using complex functions to model physical things you use the real part of the complex function as the model of what is happening. By this we mean:

$$ \text{Measured  current} =  \text{Real part of}\Big(I(t) \Big) = \text{Real}\Big( I_0 \cos(\omega t) + I_0 \sin(\omega t) \, i  \Big)  $$

The real part of this complex function will be the $\cos$ part, so we have:

$$ \text{Measured  current} = I_0 \cos{\omega t}  $$

Now, suppose you have a resistor, inductor, and capacitor in series circuit. The voltage used by the resistor, inductor, and capacitor can be found via:

$$ V_R(t) = I(t) R \hspace{2cm} V_L(t) = L \frac{dI(t)}{dt}  \hspace{2cm}  V_C(t) = \frac{1}{C} Q$$

where $R$ is the resistance of the resistor, $L$ is the inductance of the inductor, $C$ is the capacitance of the capacitor, and $Q$ is the charge built up on the capacitor. 

Let's consider the resistor as an example of what this homework is going to be about. Given the AC current above, the voltage used by the inductor will be:

$$ V_L(t)   = L \frac{d I(t)}{dt} =  L \frac{d }{dt} \left( I_0 \cos(\omega t) + I_0 \sin(\omega t) \, i  \right) =  -\omega L I_0 \sin(\omega t) + \omega L  I_0 \cos(\omega t) \, i  $$

Now, we can do a little trick here. We can rewrite the sine and cosine in terms of each other using the following trigonometric identity:

$$ \cos(\phi + \tfrac{\pi}{2}) =  -\sin(\phi) \qquad \sin(\phi + \tfrac{\pi}{2})  = \cos(\phi)   $$

These allow us to write the voltage across the inductor as:

$$ V_L(t)   =  \omega L  I_0 \cos(\omega t + \tfrac{\pi}{2}) + \omega L I_0 \sin(\omega t + \tfrac{\pi}{2}) \, i  $$

Notice that this is just the old current rotated by $\pi/2$. Comparing the phases of the voltage across the inductor and the current passing through it, we see that the phase of the voltage is $\omega t + \tfrac{\pi}{2}$ and the phase for the current is $\omega t$. If we plot these we would have something like the following (just considering the phase not the magnitudes):

<img
  src="{{ '/courses/math-methods/images/lec07/MMLec07Fig3.png' | relative_url }}"
  alt="The image shows a phasor diagram with horizontal and vertical axes labeled “Real” and “Imag,” respectively. At the origin, two arrows extend outward representing time-dependent quantities. One arrow points into the upper-right quadrant and is labeled “I of t,” indicating the current phasor. A second arrow points into the upper-left quadrant and is labeled “V L of t,” indicating the inductor voltage phasor. An arc at the origin indicates the angle between the current phasor and the horizontal real axis, labeled “omega t.” A second arc shows a ninety-degree positive phase shift between the voltage and current phasors, labeled “plus pi over two."
  style="display:block; margin:1.5rem auto; max-width:400px; width:30%;">

Notice, as these complex vectors (called phasors in in AC circuits) move around the complex plane in a counterclockwise manner $V_L(t)$ will be ahead of $I(t)$. That is, we say $V_L$ leads $I$. 



### Quantum Mechanics

Consider a spin-$\tfrac{1}{2}$ particle in a quantum system. The state of the particle can be represented by a 2-component complex vector, and we want to apply a unitary rotation to this quantum state using the Pauli spin matrices,

$$
\mathbf{\sigma_x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \qquad \mathbf{\sigma_y} = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \qquad \mathbf{\sigma_z} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

which are Hermitian and unitary.


Suppose the initial Quantum State is the particle was given as:

$$
\vec{\psi} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

In typical quantum notation, this represents the spin-up state $ \vert \uparrow\ \rangle $ along the $ z $-axis.

Let's apply a rotation to this state. This is done using to quantum rotation operator given as:

$$
\mathbf{R}(\theta) = \cos(\tfrac{\theta}{2}) \, \mathbf{I}  + i \sin(\tfrac{\theta}{2})
$$

There is an extra 1/2 in the quantum rotation operator for reasons you will investigate in a quantum mechanics course. Also, notice that the definition we have for the matrix form of $\mathbf{i}$ is the same as $\mathbf{\sigma_y}$.  For $ \theta = \frac{\pi}{2} $ (90-degree rotation), the rotation matrix becomes:

$$
R\left( \frac{\pi}{2} \right) = \cos\left( \frac{\pi}{4} \right) I - i \sin\left( \frac{\pi}{4} \right) \sigma_y
$$

and simplifies to:

$$
R\left( \frac{\pi}{2} \right) = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -i \\ i & 1 \end{bmatrix}
$$

Applying this rotation matrix to the initial state:

$$
\vec{\psi}' = R\left( \frac{\pi}{2} \right) \vec{\psi} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -i \\ i & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

Performing the matrix multiplication:

$$
\vec{\psi}' = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ i \end{bmatrix}
$$

Now our spin-$\tfrac{1}{2}$ particle has a nonzero spin in the up and down states! 










## Problem:

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.

a) Consider the following matrices:

$$
A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix} \qquad B = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \qquad C = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \qquad D = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

For each matrix, classify the matrix according to the following properties:

- **Unitary**: A matrix $ M $ is unitary if $ M^\dagger M = I $, where $ M^\dagger $ is the conjugate transpose of $ M $ and $ I $ is the identity matrix.  
- **Hermitian**: A matrix $ M $ is Hermitian if $ M^\dagger = M $.  
- **Real**: A matrix is real if all of its entries are real numbers.  
- **Diagonal**: A matrix is diagonal if all off-diagonal elements are zero.  
- **Orthogonal**: A matrix is orthogonal if $ M^T M = I $, where $ M^T $ is the transpose of $ M $.  


b) Now, consider the following Pauli spin matrices:

$$
\sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad \sigma_y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}, \quad \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

Classify these matrices using the same classifications as above.













