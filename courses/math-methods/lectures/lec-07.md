---
layout: default
title: Mathematical Methods - Lecture 07
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 7
---

# Lecture 07 – Complex Numbers, Matrices, and a New Representation
	


## Complex Numbers as 2-Dimensional Coordinates


Complex numbers naturally arise when solving for the roots of polynomial functions (like $x^2 + 1 = 0$ or something like that). By defining the imaginary unit $ i $ as $ i = \sqrt{-1} $, we can represent any complex number $ z $ in the following form: 

$$
z = a + bi
$$

where $ a $ and $ b $ are real numbers. As a result of this simple definition, we have unlocked the whole realm of mathematics called complex algebra. 

To be begin this discussion, tt is particularly convenient to think of complex numbers as points in a Cartesian-like coordinate system, where the horizontal axis represents the real part of $ z $, $ a $, and the vertical axis represents the imaginary part, $ b $. This geometric interpretation provides an intuitive way to visualize and work with complex numbers.


<div class="two-column">

<div class="column">

<img
  src="{{ '/courses/math-methods/images/lec07/MMLec07Fig1.png' | relative_url }}"
  alt="The image shows a standard coordinate plane with a horizontal axis labeled “Real” and a vertical axis labeled “Imag.” A point representing the complex number is plotted in the first quadrant. From this point, a dashed vertical line drops straight down to meet the real axis at a positive value labeled “a,” and a dashed horizontal line extends leftward to meet the imaginary axis at a positive value labeled “b.” The point itself is labeled “a plus b i.” The axes are drawn with arrows indicating positive direction."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">


</div>

<div class="column">

<img
  src="{{ '/courses/math-methods/images/lec07/MMLec07Fig2.png' | relative_url }}"
  alt="The image shows the same coordinate plane with a horizontal axis labeled “Real” and a vertical axis labeled “Imag.” A point representing the same complex number lies in the first quadrant. A dashed line connects the origin to the point, representing the radius in polar form. An angle at the origin, measured from the positive real axis up to the dashed line, indicates the direction of the point. This angle is labeled “theta,” and the dashed radius line is labeled “r.” The point itself is labeled “a plus b i."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

</div>

</div>





Notice that we can express the complex number $z$ in polar coordinates by taking $ r = \sqrt{a^2 + b^2} $ and $ \theta = \tan^{-1}\left(\frac{b}{a}\right) $, where $ r $ is called the **magnitude** or **modulus** of $ z $, and $ \theta $ is called the **phase** or **argument** of $ z $. Additionally, we can write: 

$$
a = r \cos(\theta) \qquad \text{and} \qquad b = r \sin(\theta)
$$

which allows us to express $ z $ as:

$$
z = a + i b = r \cos(\theta) + i r \sin(\theta) = r \left( \cos(\theta) + i \sin(\theta) \right).
$$









## Complex Numbers as Rotations

Now, where have we seen something like this before? Before jumping directly to an answer, let's motivate it a little. In Lecture 04, we discussed how dot products can be used to find the angle between two vectors. For example, consider one vector as the rectangular representation of the complex number above and another as a unit vector directed along the real axis:

$$
\vec{v} = a \,\widehat{\text{real}} + b \,\widehat{\text{imag}} \qquad \text{and} \qquad \widehat{\text{real}} = \widehat{\text{real}}
$$

To find the angle between these vectors, we calculate:

$$
\vec{v} \cdot \widehat{\text{real}} = \vert \vec{v} \vert   \vert \widehat{\text{real}} \vert  \cos(\theta)
$$

Substituting $ \vec{v} \cdot \widehat{\text{real}} = a $,  $  \vert \widehat{\text{real}} \vert  = 1 $, and $  \vert \vec{v} \vert  = r $, we find:

$$
a = r \cos(\theta) \implies \cos(\theta) = \frac{a}{r}
$$

Notice that $ \cos(\theta) = \frac{a}{r} $ is precisely what we obtain from solving for $ \cos(\theta) $ using the $ a $ and $ b $ components in the polar coordinate system above.

Now, let's multiply the complex number $ z = a + b i $ by another complex number: 

$$
w = \cos(\phi) + i \sin(\phi).
$$ 

(You will see why we write $ w $ in this form shortly.) The result of this multiplication, $ z' $, will be a new complex number:

$$
z' = w z
$$

Performing the algebraic multiplication:

$$
\begin{aligned}
	z' &= (\cos(\phi) + i \sin(\phi)) (a + b i) \\
	&= a \cos(\phi) + a \sin(\phi) i + b \cos(\phi) i - b \sin(\phi) \\
	&= (a \cos(\phi) - b \sin(\phi)) + (a \sin(\phi) + b \cos(\phi)) i
\end{aligned}
$$

The angle between this complex number and the positive real axis is determined by:

$$
\vec{v}' \cdot \widehat{\text{real}} = \vert \vec{v}' \vert \vert \widehat{\text{real}} \vert  \cos(\theta') \implies a \cos(\phi) - b \sin(\phi) = \vert \vec{v}' \vert \cos(\theta') \implies \cos(\theta') = \frac{a \cos(\phi) - b \sin(\phi)}{\vert \vec{v}' \vert}.
$$

Here, the vector representation of $ z' $ is:

$$
\vec{v}' = (a \cos(\phi) - b \sin(\phi)) \, \widehat{\text{real}} + (a \sin(\phi) + b \cos(\phi)) \, \widehat{\text{imag}}.
$$

The magnitude of this vector, $ \vert \vec{v}' \vert $, can be found in the following manner:

$$
\begin{aligned}
	\vert \vec{v}' \vert^2 &= (a \cos(\phi) - b \sin(\phi))^2 + (a \sin(\phi) + b \cos(\phi))^2 \\
	&= a^2 \cos^2(\phi) - 2ab \cos(\phi)\sin(\phi) + b^2 \sin^2(\phi) + a^2 \sin^2(\phi) + 2ab \sin(\phi)\cos(\phi) + b^2 \cos^2(\phi) \\
	&= a^2 (\cos^2(\phi) + \sin^2(\phi)) + b^2 (\cos^2(\phi) + \sin^2(\phi)) \\
	&= a^2 + b^2
\end{aligned}
$$
Thus, $  \vert \vec{v}' \vert  = r $, where $ r = \sqrt{a^2 + b^2} $ is the magnitude of the original complex number $ z $.

Using this result, we can express the new angle $ \theta' $ as:

$$
\begin{aligned}
\cos(\theta') &= \frac{a \cos(\phi) - b \sin(\phi)}{r} \\
&= \frac{a}{r} \cos(\phi) - \frac{b}{r} \sin(\phi) \\
&= \cos(\theta) \cos(\phi) - \sin(\theta) \sin(\phi) \\
\cos(\theta') &= \cos(\theta + \phi)
\end{aligned}
$$

This new complex number has:  
- the same magnitude as the original complex number, and  
- an angle relative to the real axis equal to the original complex number's angle plus $ \phi $.  


**Isn't this just a rotation of the original vector by an angle $ \boldsymbol{\phi} $?**  



{% capture ex %}
Multiplying a complex number $ z $ by the complex number:  

$$
w = \cos(\phi) + \sin(\phi) \, i  
$$  

results in a rotation of $ z $ by an angle $ \phi $ in the counterclockwise direction.  
{% endcapture %}
{% include result.html content=ex %}
	








## Complex Numbers as Matrices 

We can perform a change of representation, writing the complex number $ z $ as a vector and the complex number $ w $ as a rotation matrix, as follows:  

$$
z' = w z \implies \vec{v}' = \mathbf{R}(\phi) \, \vec{v}  
$$  

We have already seen how we can rewrite a complex number as a vector:  

$$
z = a + b \, i \implies \vec{v} = a\,\,\widehat{\text{real}} + b\,\,\widehat{\text{imag}} = \begin{bmatrix}  
	a \\ b  
\end{bmatrix}  
$$  

But how can we represent the complex number $ w $ as a matrix? Let’s consider what the corresponding rotation matrix would look like, since we already know what a 2-dimensional rotation matrix looks like:  

$$
\mathbf{R}(\phi) = \begin{bmatrix}  
	\cos(\phi) & -\sin(\phi) \\  
	\sin(\phi) & \cos(\phi)  
\end{bmatrix}  
$$  

Separating this into its sine and cosine parts gives:  

$$
\mathbf{R}(\phi) = \begin{bmatrix}  
	\cos(\phi) & 0 \\  
	0 & \cos(\phi)  
\end{bmatrix} + \begin{bmatrix}  
	0 & -\sin(\phi) \\  
	\sin(\phi) & 0  
\end{bmatrix} = \cos(\phi) \begin{bmatrix}  
	1 & 0 \\  
	0 & 1  
\end{bmatrix} + \sin(\phi)\begin{bmatrix}  
	0 & -1 \\  
	1 & 0  
\end{bmatrix}  
$$  

Comparing this to $ w $:  

$$
w = \cos(\phi) + \sin(\phi) \, i = \cos(\phi)\cdot 1 + \sin(\phi)\cdot i  
$$  

suggests the following equivalences:  

$$
1 \rightarrow \begin{bmatrix}  
	1 & 0 \\  
	0 & 1  
\end{bmatrix}  
$$  

where the number $ 1 $ goes over as the identity matrix, which makes sense. Similarly,  

$$
i \rightarrow \begin{bmatrix}  
	0 & -1 \\  
	1 & 0  
\end{bmatrix}  
$$  
the complex number $ i $ goes over to this antisymmetric matrix. This allows us to write $ w $ in matrix form as a rotation matrix:  

$$
\mathbf{w} = \begin{bmatrix}  
	\cos(\phi) & -\sin(\phi) \\  
	\sin(\phi) & \cos(\phi)  
\end{bmatrix}  
$$  

which makes sense, since as we saw earlier $ w $ acts like a rotation on the complex number $ z $.  

This mapping of $1$ and $i$ allows us to write any complex number as a matrix, for example:

$$ 
z = a + i b \implies \mathbf{z} = a \begin{bmatrix}
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

Let's take the representation and see what the matrix operations we are already familiar with mean in terms of complex numbers and their 2-dimensional geometric:

- **Determinant**: The determinant of $\mathbf{z}$:

	$$ \text{det}(\mathbf{z}) = \begin{vmatrix}
	a & -b \\
	b & a
	\end{vmatrix} = a^2 - (-b^2) = a^2 + b^2 $$

	represents the magnitude of the complex number $z$. 


- **Transpose**: The transpose of $\mathbf{z}$:

	$$ \mathbf{z}^\text{T} = \begin{bmatrix}
	a & b \\
	-b & a
	\end{bmatrix} = a \begin{bmatrix}
	1 & 0 \\
	0 & 1
	\end{bmatrix} - b \begin{bmatrix}
	0 & -1 \\
	1 & 0
	\end{bmatrix} = a\mathbf{I} - b\mathbf{i} = \mathbf{z}^* $$

	appears to be the complex conjugation of the complex number $z$.


- **Inverse**: The inverse of $\mathbf{z}$:

	$$ \mathbf{z}^{-1} = \frac{1}{\det(\mathbf{z})} \begin{bmatrix}
	a & b \\
	-b & a
	\end{bmatrix} = \frac{1}{a^2+b^2} \mathbf{z}^\text{T} = \frac{\mathbf{z}^*}{a^2+b^2}   $$

	looks to be the simplification of $1/z$, removing and trace of $i$ from the denominator:

	$$ z^{-1} =  \frac{1}{z} = \frac{1}{a + i b} = \frac{1}{a + i b}\frac{a - i b}{a - i b} = \frac{z^*}{a^2 + b^2}  $$ 


- **Symmetric and Antisymmetric Decomposition**:
	- The symmetric component of $\mathbf{z}$:

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

	- The antisymmetric component of $\mathbf{z}$:

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

Writing $\mathbf{z}$ in terms of its symmetric and antisymmetric components is identical to writing a complex number in terms of its real and imaginary parts. 









## Complex Matrices

A **complex matrix** is a matrix whose elements can be complex numbers. These matrices generalize many properties of real matrices while enabling operations and transformations in spaces where both magnitude and phase (angle) are important.

Similar to real matrices, complex matrices support addition, subtraction, scalar multiplication, and matrix multiplication in the same way as real matrices. The underlying arithmetic is governed by the rules of complex numbers.

A key difference for complex matrices is the availability of additional operations. For example, you can compute the complex conjugate of the elements of the matrix:

$$
\mathbf{A} = \begin{bmatrix}
	a_{11} & a_{12}\\
	a_{21} & a_{22}
\end{bmatrix} \qquad \mathbf{A}^* = \begin{bmatrix}
	a_{11}^* & a_{12}^*\\
	a_{21}^* & a_{22}^*
\end{bmatrix}
$$

where $ \mathbf{A} $ is a complex matrix, meaning its elements may be complex numbers. 


Additionally, we can compute the transpose of a complex matrix as usual:

$$
\mathbf{A} = \begin{bmatrix}
	a_{11} & a_{12}\\
	a_{21} & a_{22}
\end{bmatrix} \qquad \mathbf{A}^\text{T} = \begin{bmatrix}
	a_{11} & a_{21}\\
	a_{12} & a_{22}
\end{bmatrix}
$$

A more powerful operation is the combination of the complex conjugate and the transpose, known as the **Hermitian adjoint** (or simply the **Hermitian**):

$$
\mathbf{A} = \begin{bmatrix}
	a_{11} & a_{12}\\
	a_{21} & a_{22}
\end{bmatrix} \qquad \mathbf{A}^\dagger = \begin{bmatrix}
	a_{11}^* & a_{21}^*\\
	a_{12}^* & a_{22}^*
\end{bmatrix}
$$


These new operations lead to important classifications of complex matrices:

- **Hermitian Matrices**: Matrices that satisfy $ \mathbf{A} = \mathbf{A}^\dagger $. These are the complex analogs of symmetric matrices.
- **Unitary Matrices**: Matrices that satisfy $ \mathbf{U}^\dagger \mathbf{U} = \mathbf{I} $, representing length-preserving transformations (such as rotations in complex space).
- **Normal Matrices**: Matrices that satisfy $ \mathbf{A}^\dagger \mathbf{A} = \mathbf{A} \mathbf{A}^\dagger $, a condition that generalizes diagonalizability. 
	- Since a Hermitian matrix is equal to its own Hermitian adjoint ($ \mathbf{A} = \mathbf{A}^\dagger $), the Normal matrix condition ($ \mathbf{A}^\dagger \mathbf{A} = \mathbf{A} \mathbf{A}^\dagger $) will be automatically satisfied. Thus, all Hermitian matrices are also Normal, but not all Normal matrices are Hermitian. In other words, Hermitian matrices form a subset of the broader category of Normal matrices.


{% capture ex %}
Suppose we are given the following matrices:

$$
\mathbf{A} = \begin{bmatrix}
	2 & i \\
	-i & 3
\end{bmatrix} \qquad 
\mathbf{B} = \frac{1}{\sqrt{2}} \begin{bmatrix}
	1 & i \\
	-i & 1
\end{bmatrix} \qquad 
\mathbf{C} = \begin{bmatrix}
	0 & 1 + i \\
	1 - i & 0
\end{bmatrix}
$$

and we are tasked with check to see if they are Hermitian, Unitary, and/or Normal matrices. Let's check by check if each matrix has the property that defined each classification.

**Checking Hermitian Property ($ \mathbf{A} = \mathbf{A}^\dagger $)**

- **Matrix $\mathbf{A}$**:  
The conjugate transpose is:

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
The conjugate transpose is:

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
The conjugate transpose is:

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

When thinking about complex numbers and their representation in 2-dimensional space, one might begin to wonder what other dimensional spaces can be expressed using complex--or complex-like-- numbers. In 1842 William Rowan Hamilton stumbled upon an idea that opened complex-like numbers up to a 4-dimensional space (1 real and 3 imaginary). Numbers written in this space are called quaternions and are used extensively throughout physics, especially in quantum mechanics and quantum electrodynamics.

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
ij = k, \quad ji = -k, \qquad jk = i, \quad kj = -i, \qquad ki = j, \quad ik = -j
$$

Notice that these relations resemble the cross product, and the letters used ($i, j, k$) are the same as the ones used for the unit vectors for the $x$-, $y$-, and $z$-axes. This is not a coincidence. Quaternions were developed before vectors and their operations were formally introduced. In fact, vectors and their operations emerged as an attempt to simplify the complicated mathematics involved in quaternion manipulations. This development was done in the mid-1880's by Oliver Heaviside, Hermann von Helmholtz and Josiah Willard Gibbs, some names you may recognize from other physics courses. 

Similar to how complex numbers could be represented as a couple of $2\times 2$ real matrices, quaternions can be represented as four $2 \times 2$ complex matrices. You can begin by assuming you have four arbitrary $2\times 2$ matrices that obey the relations between $i, j, k$ indicated above. Working through all of those relations and demanding that each resulting matrix have a determinant of 1, you can find what each of the four matrices needs to be. Without having to go through all of that busy work, we can write the basis elements of quaternions as:

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

We could also have decided to represent these matrices using $4\times 4$ matrices in a full 4-dimensional space. This is slightly more difficult to get, but the following matrices are what come about:

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

Quaternions and their matrix representations, particularly through the $SU(2)$ group, have a wide range of applications in both physics. In quantum mechanics, the $SU(2)$ group plays a crucial role in the description of spin-$\tfrac{1}{2}$ particles. The Pauli matrices, which are described by elements of $SU(2)$, provide a matrix representation for spin operators, allowing for the mathematical treatment of spin in quantum systems. This is particularly useful for understanding quantum states, spin precession, and the behavior of particles in magnetic fields.

In addition, $SU(2)$ is fundamental in the theory of angular momentum, where it helps describe rotations in quantum systems. It serves as the mathematical foundation for spin-orbit coupling in atomic physics and is deeply involved in the study of quantum entanglement and quantum computing, where operations on qubits are often represented using $SU(2)$ matrices.

Outside of quantum mechanics, quaternion representations also find applications in computer graphics, where they are used to represent 3D rotations. Unlike traditional matrix methods, quaternions avoid some of the numerical instability and other issues, providing more efficient and stable ways to interpolate rotations (e.g., in animations or robotics). Moreover, quaternions are employed in various fields like control theory, where their properties are leveraged in algorithms for rotation-based computations, and in the description of certain types of wave phenomena.
















## Application:

### Electronics

One place you see complex numbers in action is in when working with AC (Alternating Current) circuits. For example, in an AC circuit, the current passing through the circuit can be given as a complex number in the following manner:

$$ I(t) = I_0 w(t) =  I_0 (\cos(\omega t) + \sin(\omega t) \, i) =  I_0 \cos(\omega t) + I_0 \sin(\omega t) \, i $$

where $w(t)$ is the complex number the resulted in a rotation by $\omega t$ we saw previously (where the angle was written as $\phi$ instead of $\omega t$).

Now, you may panic and say, ``But current is not a complex thing! How is this allowed?" and you would be right to complain. To clear this up, it is generally an unsaid rule when using complex functions to model real, physical things as the real part of the complex solution. By this we mean:

$$ \text{Measured  current} =  \text{Real part of}\left(I(t) \right) = \text{Real}\left( I_0 \cos(\omega t) + I_0 \sin(\omega t) \, i  \right)  $$

The real part of the complex exponential is $\cos$, so we have:

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
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

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

In typical quantum notation, this represents the spin-up state $ \vert \uparrow \rangle $ along the $ z $-axis.

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













