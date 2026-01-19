---
layout: default
title: Mathematical Methods - Lecture 03
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 3
---

# Lecture 03 -- Rotations and Coordinate Transformations

As we've discussed, matrices can represent a wide variety of coordinate transformations: rotations, reflections, rescalings, and more. But we haven’t yet looked closely at how these transformations actually appear in matrix form, or how to properly interpret them. This lecture fills in those details.

## Matrix Transformations

When we think of a matrix as a transformation, we think of it as acting on some original vector and transforming it into a new one:

$$
\mathbf{T} \vec{r} = \vec{R}
$$

Here, the matrix $\mathbf{T}$ transforms the original vector $\vec{r}$ into a new vector $\vec{R}$. Notice that the transformation acts on the vector **to its right**. That’s not just a formatting choice, it’s built into how matrix multiplication has been defined.

If we want to transform a column vector (an $n \times 1$ matrix) into another column vector (also $n \times 1$), we need to multiply it by an $n \times n$ matrix **on the left**. That’s the only way the dimensions work out correctly. You should try verifying this: write out the shapes of each object and check that trying to multiply from the right like $\vec{r} \mathbf{T}$ doesn’t make sense if $\vec{r}$ is a column vector.

The convention of placing the transformation matrix on the left isn’t arbitrary, it’s mathematically required based on the structure of matrix multiplication and the dimensions of the vectors and matrices involved.



### Right to Left is the Order of Operations

Transformtion matrices are applied via matrix multiplication and recall matrix multiplication has the rule that order matters. This means, when applying multiple transformations to an object, the order we write these transformations matters; the order transformations are being applied is read from **right to left**. To see why this is the case, consider the following.

Suppose you start with a position vector $\vec{r}$ and apply a transformation matrix $\mathbf{T}_1$. You get the new vector $\vec{r}_1$ in the following manner:

$$
\mathbf{T}_1 \vec{r} = \vec{r}_1
$$

Suppose you then apply a second transformation $\mathbf{T}_2$ to that:

$$
\mathbf{T}_2 \vec{r}_1 = \vec{r}_2
$$

giving you $\vec{r}_2$ as a result.

If we substitute the first equation into the second:

$$
\mathbf{T}_2 \big(\mathbf{T}_1 \vec{r}\big) = \vec{r}_2
$$

Because matrix multiplication is **associative** (but not commutative!), we can write:

$$
\big(\mathbf{T}_2 \mathbf{T}_1\big) \vec{r} = \vec{r}_2
$$

So, even though $\mathbf{T}_2$ appears in the left-most position in the expression, it’s applied *after* $\mathbf{T}_1$. Remember, transformation matrices act to the right. This means **the transformation closest to the vector gets applied first.** Thus, transofrmations are applied from right to left. 

You can generalize this to any number of transformations:

$$
\mathbf{T}_n \cdots \mathbf{T}_2 \mathbf{T}_1 \vec{r} = \vec{R}
$$

Here, $\mathbf{T}_1$ happens first, then $\mathbf{T}_2$, and so on until $\mathbf{T}_n$ is applied last.

{% capture ex %}
The order matrix transformations are applied is from the right to left.
{% endcapture %}
{% include result.html content=ex %}

This right-to-left order isn’t just a mathematical technicality, it has real physical consequences. In physics, the sequence in which you apply rotations, reflections, and other transformations can dramatically change the result. For instance, in Quantum Field Theory, when writing down how a neutron decays into a proton, an electron, and an anti-electron neutrino, the math is read from right to left matching the physical sequence of interactions.

Grasping this structure is especially important in fields like 3D coordinate transformations and quantum mechanics, where multiple matrices act on a system in a precise order. Getting that order wrong doesn’t just mess up the calculation, it changes what the math *means* physically.








## Rotations

The most straightforward way to find the matrix representation of a rotation about the $z$-acis is to examine how the unit vectors in the $xy$-plane are transformed under rotation. Imagine the original $xy$-coordinate system, and then draw a new coordinate system that has been rotated counterclockwise by an angle $\theta$ about the $z$-axis (pointing out of the page). The setup would look like this:

<img
  src="{{ '/courses/math-methods/images/lec03/rotatedcorrdinates.png' | relative_url }}"
  alt="The image shows two sets of coordinate axes that share the same origin. The original axes are drawn in black, with the horizontal axis pointing to the right and labeled i-hat, and the vertical axis pointing upward and labeled j-hat. A rotated coordinate system is drawn in red. The red i-prime unit vector extends from the origin at an angle above the black horizontal axis. The red j-prime unit vector extends from the origin at an angle above the negative side of the black horizontal axis. Both red vectors form equal angles, labeled theta, with their adjacent black axes."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

Using trigonometry, and remembering that unit vectors have length 1, we can determine that the $x$-component of $\widehat{i}'$ is $\cos(\theta)$ and the $y$-component is $\sin(\theta)$. We can do the same thing for $\widehat{j}'$, and the results can be written as:

$$
\begin{aligned}
	\cos(\theta) \,\widehat{i} + \sin(\theta) \,\widehat{j} = \widehat{i}'  \\
	-\sin(\theta) \,\widehat{i} + \cos(\theta) \,\widehat{j} = \widehat{j}'
\end{aligned}
$$

Check the $\widehat{j}'$ expression to make sure you agree!

Now, using the standard (and simplest) matrix representations for the unit vectors:

$$
\widehat{i} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
\qquad \text{and} \qquad
\widehat{j} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

we find:

$$
 \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix} = \widehat{i}'
\qquad \text{and} \qquad
\begin{bmatrix} -\sin(\theta) \\ \cos(\theta) \end{bmatrix} = \widehat{j}' 
$$

Now, we have the original basis vectors $\widehat{i}$ and $\widehat{j}$ and the new, rotated basis vectors $\widehat{i}'$ and $\widehat{j}'$. Our goal is to find the rotation matrix that transforms the original vectors into the rotated ones:

$$
\begin{aligned}
	\mathbf{R}_z(\theta) \, \widehat{i} &= \widehat{i}' \\
	\mathbf{R}_z(\theta) \, \widehat{j} &= \widehat{j}'
\end{aligned}
$$

where $\mathbf{R}_z(\theta)$ is the matrix that performs a counterclockwise rotation about the $z$-axis by an angle $\theta$.

Since this matrix acts on a $2 \times 1$ vector and returns another $2 \times 1$ vector, it must be a $2 \times 2$ matrix. We can write its general form as:

$$
\mathbf{R} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

Our task is to determine the values of $a$, $b$, $c$, and $d$ that ensure this matrix rotates $\widehat{i}$ and $\widehat{j}$ into $\widehat{i}'$ and $\widehat{j}'$, respectively.

To do this we can start by applying $\mathbf{R}$ to $\widehat{i}$:

$$
\mathbf{R} \widehat{i} = \widehat{i}' \quad \Rightarrow \quad
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
\begin{bmatrix} 1 \\ 0 \end{bmatrix} =
\begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix}
\quad \Rightarrow \quad
\begin{bmatrix} a \\ c \end{bmatrix} = \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix}
$$

So, $a = \cos(\theta)$ and $c = \sin(\theta)$.

Now do the same for $\widehat{j}$:

$$
\mathbf{R} \widehat{j} = \widehat{j}' \quad \Rightarrow \quad
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
\begin{bmatrix} 0 \\ 1 \end{bmatrix} =
\begin{bmatrix} -\sin(\theta) \\ \cos(\theta) \end{bmatrix}
\quad \Rightarrow \quad
\begin{bmatrix} b \\ d \end{bmatrix} = \begin{bmatrix} -\sin(\theta) \\ \cos(\theta) \end{bmatrix}
$$

giving $b = -\sin(\theta)$ and $d = \cos(\theta)$.

{% capture ex %}
The 2x2 rotation matrix $\mathbf{R}(\theta)$ that rotates a vector by an angle $\theta$ counterclockwise about the $z$-axis is given by:

$$
\mathbf{R}(\theta) = \begin{bmatrix}
	\cos(\theta) & -\sin(\theta) \\
	\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

Please note, this matrix was designed to transform objects written in the original basis, in terms of $\widehat{i}$ and $\widehat{j}$, into their new representation in the rotated basis, $\widehat{i}'$ and $\widehat{j}'$.
{% endcapture %}
{% include result.html content=ex %}

{% capture ex %}
Suppose we wish to rotate the vector $(1,1)$ by 45 degrees about the $z$-axis. The rotation matrix and vector are:

$$
\mathbf{R}(45^\circ) = \begin{bmatrix}
	\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
	\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
\end{bmatrix}
\qquad
\vec{r} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

Applying the rotation:

$$
\vec{r}' = \mathbf{R}(45^\circ) \vec{r} =
\begin{bmatrix} 0 \\ \tfrac{2}{\sqrt{2}} \end{bmatrix}
$$

This makes sense! The vector $(1,1)$ lies at a 45-degree angle from the $x$-axis. Rotating it another 45 degrees moves it directly along the $y$-axis, making its $x$-component zero. The vector’s magnitude is $|\vec{r}| = \sqrt{2}$, which matches the new $y$-component after simplifying $\tfrac{2}{\sqrt{2}}$.
{% endcapture %}
{% include example.html content=ex %}

To extend this to 3D, we can build a rotation matrix about the $z$-axis that leaves the $z$-component unchanged:

$$
\mathbf{R}_z(\theta) = \begin{bmatrix}
	\cos(\theta) & -\sin(\theta) & 0 \\
	\sin(\theta) & \cos(\theta) & 0 \\
	0 & 0 & 1
\end{bmatrix}
$$

You can derive the rotation matrices for the $x$- and $y$-axes the same way:

**Rotation about the $x$-axis:**

$$
\mathbf{R}_x(\alpha) = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & \cos(\alpha) & -\sin(\alpha) \\
	0 & \sin(\alpha) & \cos(\alpha)
\end{bmatrix}
$$

**Rotation about the $y$-axis:**

$$
\mathbf{R}_y(\beta) = \begin{bmatrix}
	\cos(\beta) & 0 & \sin(\beta) \\
	0 & 1 & 0 \\
	-\sin(\beta) & 0 & \cos(\beta)
\end{bmatrix}
$$

{% capture ex %}
Suppose we rotate the vector $(1, 0, 0)$ by 90 degrees about the $z$-axis, then by 90 degrees about the $x$-axis.

**First Rotation:**

$$
\mathbf{R}_z(90^\circ) = \begin{bmatrix}
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{bmatrix}
\quad
\vec{r} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
\quad
\Rightarrow \quad
\mathbf{R}_z \vec{r} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

**Second Rotation:**

$$
\mathbf{R}_x(90^\circ) = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & 0 & -1 \\
	0 & 1 & 0
\end{bmatrix}
\quad
\Rightarrow \quad
\vec{r}\,' = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

If you reverse the order (first rotate about $x$, then about $z$), the result is different:

$$
\mathbf{R}_x(90^\circ) \vec{r} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
\quad
\Rightarrow \quad
\mathbf{R}_z(90^\circ) \vec{r} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

The takeaway: **the order of matrix multiplication matters**. This is called **non-commutativity**.
{% endcapture %}
{% include example.html content=ex %}

The fact that operations can be represented by matrices—and that the order of matrix multiplication often matters—is crucial for understanding Quantum Mechanics. Heisenberg was among the first to recognize that the order in which the position and momentum operators act on a quantum state matters. Specifically, he discovered that position and momentum do **not** commute. This non-commutativity, expressed in the canonical commutation relation

$$
[\widehat{x}, \widehat{p}_x] = \widehat{x} \widehat{p}_x - \widehat{p}_x \widehat{x} = i \hbar
$$

was a key insight that led Heisenberg, with contributions from Born and Jordan, to develop *Matrix Mechanics*, one of the earliest formulations of Quantum Mechanics.

At the time, matrices were relatively new to physics, and it was Jordan—familiar with recent mathematical developments—who introduced Heisenberg to matrices and helped guide the mathematical formulation. This collaboration laid the foundation for much of modern quantum theory.









## Rotations

The most straightforward way to find the rotation matrix is to examine how the unit vectors in the $xy$-plane are transformed under rotation. Imagine the original $xy$-coordinate system, and then draw a new coordinate system that has been rotated counterclockwise by an angle $\theta$ about the $z$-axis (pointing out of the page). The setup would look like this:


<img
  src="{{ '/courses/math-methods/images/lec03/rotatedcorrdinates.png' | relative_url }}"
  alt="The image shows two sets of coordinate axes that share the same origin. The original axes are drawn in black, with the horizontal axis pointing to the right and labeled i-hat, and the vertical axis pointing upward and labeled j-hat. A rotated coordinate system is drawn in red. The red i-prime unit vector extends from the origin at an angle above the black horizontal axis. The red j-prime unit vector extends from the origin at an angle above the negative side of the black horizontal axis. Both red vectors form equal angles, labeled theta, with their adjacent black axes."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

Using trigonometry, and the fact unit vectors have a length of $1$, we can see that the $x$-component of $\widehat{i}'$ is give as $\cos(\theta)$ and the $y$-component is given by $\sin(\theta)$. We can do the same thing $\widehat{j}'$, and the results can be written as:

$$ \begin{aligned}
	\cos(\theta) \,\widehat{i} + \sin(\theta) \,\widehat{j} = \widehat{i}'  \\
	-\sin(\theta) \,\widehat{i} + \cos(\theta) \,\widehat{j} = \widehat{j}' \\
\end{aligned} $$

Check the $\widehat{j}'$ to make sure you agree! 

Now, if we use the typical, and simplest, matrix representation for the unit vectors:
$$\widehat{i} = \begin{bmatrix}
	1 \\ 0
\end{bmatrix} \qquad \text{and} \qquad \widehat{j} = \begin{bmatrix}
0 \\ 1
\end{bmatrix} $$
We are left with the following result:
$$ 
	\begin{bmatrix}
		\cos(\theta) \\ \sin(\theta)
	\end{bmatrix} = \widehat{i}'  
	\qquad \text{and} \qquad
	\begin{bmatrix}
		-\sin(\theta) \\ \cos(\theta)
	\end{bmatrix} = \widehat{j}' 
$$

So, we need a matrix that will rotate $\widehat{i}$ to $\widehat{i}'$, and similarly $\widehat{j}$ to $\widehat{j}'$. Suppose we have the following matrix:

$$  \mathbf{R} = \begin{bmatrix}
	a & b \\ c& d
\end{bmatrix} $$

and we wish to force it to be the rotation matrix. This means we just need to find what the elements have to be to get the correct rotations of $\widehat{i}$ and $\widehat{j}$. 

For instance, we will require the following to be true:

$$ \mathbf{R} \, \widehat{i} = \widehat{i}' \quad \implies \quad  \begin{bmatrix}
	a & b \\ c& d
\end{bmatrix}  \begin{bmatrix}
1 \\ 0
\end{bmatrix} = \begin{bmatrix}
\cos(\theta) \\ \sin(\theta)
\end{bmatrix} \quad \implies \quad  \begin{bmatrix}
a \\ c
\end{bmatrix} = \begin{bmatrix}
\cos(\theta) \\ \sin(\theta)
\end{bmatrix}  $$
this means $a = \cos(\theta)$ and $c = \sin(\theta)$. 

Similarly, we enforce:
$$ \mathbf{R} \, \widehat{j} = \widehat{j}' \quad \implies \quad \begin{bmatrix}
	a & b \\ c& d
\end{bmatrix}  \begin{bmatrix}
	0 \\ 1
\end{bmatrix} = \begin{bmatrix}
	-\sin(\theta) \\ \cos(\theta)
\end{bmatrix} \quad \implies \quad  \begin{bmatrix}
	b \\ d
\end{bmatrix} = \begin{bmatrix}
	-\sin(\theta) \\ \cos(\theta)
\end{bmatrix}  $$
which tells us $b = -\sin(\theta)$ and $d = \cos(\theta)$. 


{% capture ex %}
The 2x2 rotation matrix $\mathbf{R}(\theta)$ that rotates a vector by an angle $\theta$ counterclockwise about the $z$-axis is given by:
$$
\mathbf{R}(\theta) = \begin{bmatrix}
	\cos(\theta) & -\sin(\theta) \\
	\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

Please note, this matrix was designed to transform objects written in the original basis, in terms of $\widehat{i}$ and $\widehat{j}$, into their new representation in the rotated basis, $\widehat{i}'$ and $\widehat{j}'$.
{% endcapture %}
{% include result.html content=ex %}


{% capture ex %}
Suppose we wish to rotate the vector $ (1,1) $ by 45 degrees about the $ z $-axis. The rotation matrix and vector for this situation are given by:

$$
\mathbf{R}(45^\circ) = \begin{bmatrix}
	\cos(45^\circ) & -\sin(45^\circ) \\
	\sin(45^\circ) & \cos(45^\circ)
\end{bmatrix} = \begin{bmatrix}
	\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
	\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
\end{bmatrix} \qquad \vec{r} = \begin{bmatrix}
	1 \\ 1
\end{bmatrix}
$$

Applying the rotation matrix to the vector, we have:

$$
\vec{r}\,' = \mathbf{R}(45^\circ) \, \vec{r} = \begin{bmatrix}
	\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
	\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
\end{bmatrix} \begin{bmatrix}
	1 \\ 1
\end{bmatrix} = \begin{bmatrix}
	\tfrac{1}{\sqrt{2}} + (-\tfrac{1}{\sqrt{2}}) \\ \tfrac{1}{\sqrt{2}} + \tfrac{1}{\sqrt{2}}
\end{bmatrix} = \begin{bmatrix}
	0 \\ \tfrac{2}{\sqrt{2}}
\end{bmatrix}
$$

which makes sense! 

Why? Because the vector $ (1,1) $ is already oriented at 45 degrees above the positive $ x $-axis. Rotating it by an additional 45 degrees brings it all the way to the $ y $-axis, making the $ x $-component zero. Furthermore, the length of this vector is $ |\vec{r}| = \sqrt{(1)^2 + (1)^2} = \sqrt{2} $, which agree with the $y$-component of the new vector after simplifying $ \tfrac{2}{\sqrt{2}} $.
{% endcapture %}
{% include example.html content=ex %}

To extend this to a 3-dimensional rotation matrix, we can include an unchanged $ z $-component as follows:

$$
\mathbf{R}_z(\theta) = \begin{bmatrix}
	\cos(\theta) & -\sin(\theta) & 0 \\
	\sin(\theta) & \cos(\theta) & 0 \\
	0  & 0 & 1
\end{bmatrix}
$$

This matrix represents a rotation by an angle $ \theta $ about the $ z $-axis in three-dimensional space, leaving the $ z $-coordinate unaffected. Through a similar logic, you can get the rotations matrices for rotations about the $x$ and $y$ axes. 

For a rotation by an angle $ \alpha $ about the $ x $-axis:

$$
\mathbf{R}_x(\alpha) = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & \cos(\alpha) & -\sin(\alpha) \\
	0 & \sin(\alpha) & \cos(\alpha) 
\end{bmatrix}
$$

For a rotation by an angle $ \beta $ about the $ y $-axis:

$$
\mathbf{R}_y(\beta) = \begin{bmatrix}
	\cos(\beta) & 0 & \sin(\beta) \\
	0 & 1 & 0 \\
	-\sin(\beta) & 0 & \cos(\beta) 
\end{bmatrix}
$$

{% capture ex %}
Suppose we wish to rotate the vector $ (1,0,0) $ by 90 degrees about the $ z $-axis, followed by another 90-degree rotation about the $ x $-axis. The rotation matrices for each of these rotations, as well as the initial vector, are given by:

$$
\mathbf{R}_z(90^\circ) = \begin{bmatrix}
	\cos(90^\circ) & -\sin(90^\circ) & 0 \\
	\sin(90^\circ) & \cos(90^\circ) & 0 \\
	0 & 0 & 1
\end{bmatrix} = \begin{bmatrix}
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{bmatrix}
$$

$$
\mathbf{R}_x(90^\circ) = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & \cos(90^\circ) & -\sin(90^\circ) \\
	0 & \sin(90^\circ) & \cos(90^\circ)
\end{bmatrix} = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & 0 & -1 \\
	0 & 1 & 0
\end{bmatrix}
$$

$$
\vec{r} = \begin{bmatrix}
	1 \\ 0 \\ 0 
\end{bmatrix}
$$

To apply these rotations, we start by rotating the vector about the $ z $-axis:

$$
\mathbf{R}_z(90^\circ) \vec{r} = \begin{bmatrix}
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
	1 \\ 0 \\ 0
\end{bmatrix} = \begin{bmatrix}
	0 \\ 1 \\ 0
\end{bmatrix}
$$

Next, we apply the 90-degree rotation about the $ x $-axis to the result:

$$
\mathbf{R}_x(90^\circ) \left( \mathbf{R}_z(90^\circ) \vec{r} \right) = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & 0 & -1 \\
	0 & 1 & 0
\end{bmatrix} \begin{bmatrix}
	0 \\ 1 \\ 0
\end{bmatrix} = \begin{bmatrix}
	0 \\ 0 \\ 1
\end{bmatrix}
$$

Thus, the final vector after both rotations is:

$$
\vec{r}\,' = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

In this case, the order of operations matters. To demonstrate this, we can try performing the rotations in the reverse order: first rotating about the $ x $-axis and then about the $ z $-axis. Starting with the initial vector:

$$
\mathbf{R}_x(90^\circ) \vec{r} = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & 0 & -1 \\
	0 & 1 & 0
\end{bmatrix} \begin{bmatrix}
	1 \\ 0 \\ 0
\end{bmatrix} = \begin{bmatrix}
	1 \\ 0 \\ 0
\end{bmatrix}
$$

Now, applying the rotation about the $ z $-axis:

$$
\mathbf{R}_z(90^\circ) \left( \mathbf{R}_x(90^\circ) \vec{r} \right) = \begin{bmatrix}
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
	1 \\ 0 \\ 0
\end{bmatrix} = \begin{bmatrix}
	0 \\ 1 \\ 0
\end{bmatrix}
$$

The final result is different from our previous result, showing that the order of rotations affects the outcome. In mathematics, this property is called **non-commutativity**, meaning the sequence in which the operations, in this case rotations, are applied is important for getting the correct final result.
{% endcapture %}
{% include example.html content=ex %}

The fact that operations can be represented by matrices, and that the order of matrix multiplication often matters, is crucial for understanding Quantum Mechanics. Heisenberg was among the first to recognize that the order in which the position and momentum operators act on a quantum state matters—specifically, he discovered that position and momentum do not commute. This non-commutativity, expressed in the canonical commutation relation 
$$
[\widehat{x}, \widehat{p}_x] = \widehat{x} \widehat{p}_x - \widehat{p}_x \widehat{x} = i \hbar
$$
was a key insight that led Heisenberg, with contributions from Born and Jordan, to develop *Matrix Mechanics*, one of the earliest formulations of Quantum Mechanics. In this approach, observable quantities like position and momentum are represented by matrices, and their non-commuting nature underlies the uncertainty principle.

At the time, matrices were relatively new to physics, and it was Jordan, a mathematician familiar with recent mathematical developments, who introduced Heisenberg to matrices and guided him in this direction. This collaboration was instrumental in shaping the foundations of Quantum Mechanics.






### Properties of Rotation Matrices

The determinant of a rotation matrix is always $ +1 $. This means rotations are not capable of recreating a reflection (determinate $-1$, as we will see) no matter what rotation matrices you use. We can check our $2 \times 2$ matrix to make sure it obeys this property:

$$ \begin{vmatrix}
	\cos(\theta) & -\sin(\theta) \\
	\sin(\theta) & \cos(\theta)
\end{vmatrix} = \cos^2(\theta) - (-\sin^2(\theta)) = \cos^2(\theta)  + \sin^2(\theta) = 1 $$

In more formal terms, this property indicates that rotation matrices belong to the **special orthogonal group**, commonly denoted as $ SO(n) $, where $ n $ is the dimension of the space (e.g., $ SO(2) $ for 2D rotations and $ SO(3) $ for 3D rotations). *Special* means the matrices all ave a determinate of $+1$, and we will discuss what *orthogonal* means for matrices in Lecture 05. Since the determinant of a rotation matrix equals $ +1 $, these matrices preserve the ``handedness" or orientation of the coordinate system, ensuring that any rotation does not reflect or invert the space. The special orthogonal groups are vital in the study of particle physics and quantum field theories, as rotations represent fundamental symmetries of nature.


Another key property of rotation matrices, that we kind of showed above, is that successive rotations can be represented by the product of two or more rotation matrices. For example, if $ \mathbf{R}_1 $ and $ \mathbf{R}_2 $ are rotation matrices, then their product $ \mathbf{R} = \mathbf{R}_1 \mathbf{R}_2 $ is also a rotation matrix, representing the combined effect of the two rotations:
$$
\mathbf{R} = \mathbf{R}_1 \mathbf{R}_2
$$
It is important to note that matrix multiplication is generally not commutative, meaning $ \mathbf{R}_1 \mathbf{R}_2 \neq \mathbf{R}_2 \mathbf{R}_1 $ in most cases. Thus, the order of operations matters when applying rotations sequentially, especially in three-dimensional space (or higher). This property is particularly useful in physics for describing complex transformations. For instance, in quantum mechanics, combined rotations correspond to transformations in the system’s state space.








## Reflections

Another interesting coordinate transformation made possible using matrices is reflections. Reflections in geometry occur when you mirror a vector over one, or multiple, axes. In effect this take one of the coordinates from, say $x$ to $-x$ without changing any of the other coordinates. 

In general, a reflection matrix transforms points across a specified line in two dimensions or a plane in three dimensions. For instance, reflecting a point across the $ x $-axis in two dimensions can be represented by the following matrix:

$$
\mathbf{R}_{x} = \begin{bmatrix}
	1 & 0 \\
	0 & -1
\end{bmatrix}
$$

For example, if we apply this matrix to the vector $(2,5)$ we would get:

$$ \begin{bmatrix}
	1 & 0 \\
	0 & -1
\end{bmatrix} \begin{bmatrix}
2 \\ 5
\end{bmatrix} = \begin{bmatrix}
2 \\ -5
\end{bmatrix} $$

which has carried the $y$ value to the negative of that value without changing the $x$ value. 

In three dimensions, reflecting a point across the $ xy $-plane can be represented as:

$$
\mathbf{R}_{xy} = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & 1 & 0 \\
	0 & 0 & -1
\end{bmatrix}
$$

{% capture ex %}
Consider a situation in geometric optics where a light ray strikes a flat mirror, which lies along the $ x $-axis in a 2D plane, say attached to the ceiling (for some reason, maybe a dance party or something). Suppose a light ray initially travels along a vector directed 45 degrees above the $ x $-axis. This ray strikes the mirror and reflects causing its $ y $-component to change sign, while its $ x $-component remains the same, resulting in a reflection across the $ x $-axis--moving to the right and up to now moving to the right and down.

To model this reflection mathematically, we can represent the mirror’s effect on the light ray using a reflection matrix. In 2D, a reflection matrix across the $ x $-axis can be written as:
$$
\mathbf{R}_{\text{x}} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$
This matrix flips the sign of the $ y $-component of any vector while leaving the $ x $-component unchanged, as we saw above.

Suppose the incident light ray can be represented by the vector (the magnitude is unimportant in this example): 
$$
\vec{r}_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

To find the direction of the reflected ray, we apply the reflection matrix:
$$
\vec{r}_{\text{reflected}} = \mathbf{R}_{\text{x}} \, \vec{r}_0 = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}.
$$
The resulting vector points 45 degrees below the $ x $-axis, showing that the ray has been reflected as expected.

This example, models the action of a mirror perfectly aligned with the $ x $-axis reflecting a ray of light. In optics, this concept generalizes to more complex setups, such as reflections from angled mirrors or transformations in multi-dimensional systems, where reflection matrices are used to compute the paths and behaviors of light or other particles under reflection transformations.
{% endcapture %}
{% include example.html content=ex %}


### Properties of Reflection Matrices

A key property of reflection matrices is that they have a determinant of $ -1 $ (go ahead and check this for the two reflection matrices we wrote down previously), indicating that they change the "handedness" or orientation of the coordinate system. This is fundamentally different from rotation matrices, which preserve the orientation of the coordinate system. In particle physics, a change in the "handedness" of the system is referred to as a parity change. Reflection matrices in important for studying certain processes in particle physics which exhibit parity violation.














## Rescaling/Scaling

Rescaling (or scaling) matrices are used to change the size of geometric objects in space without altering their shape. A rescaling matrix modifies the length of vectors in one or more dimensions by multiplying them by a scaling factor. The general form of a 2D rescaling matrix is:

$$
\mathbf{S} = \begin{bmatrix}
	s_x & 0 \\
	0 & s_y
\end{bmatrix}
$$

where $ s_x $ and $ s_y $ are the scaling factors in the $ x $ and $ y $ directions, respectively. For 3D space, the scaling matrix can be expressed as:

$$
\mathbf{S} = \begin{bmatrix}
	s_x & 0 & 0 \\
	0 & s_y & 0 \\
	0 & 0 & s_z
\end{bmatrix}
$$

where $ s_z $ is the scaling factor in the $ z $ direction. 

### Properties of Rescaling/Scaling Matrices

Unlike rotation and reflection matrices, rescaling matrices can have determinants that are positive, negative, or zero, reflecting whether the transformation preserves orientation, reverses it, or collapses the space. Reflection matrices can be thought of as special cases of rescaling matrices where some axes are multiplied by $-1$, changing orientation but preserving magnitude.







## Transforming Matrices

At this point, you might be wondering: how do we reflect across axes other than the $x$- or $y$-axis? 

And your instinct might run along the lines of, “If I can rotate a mirror in real life, maybe I can just rotate the reflection matrix too!” 

That intuition is on point—but we haven’t yet talked about how to rotate a matrix itself. So let’s try to figure it out by experimenting.

Suppose we want a reflection matrix for a mirror set along the line $ y = x $, which is angled at 45$^\circ$ above the horizontal. One idea might be to take the standard reflection across the $x$-axis, $ \mathbf{R}_x $, and rotate it by 45$^\circ$ using the rotation matrix:

$$
\mathbf{R}(45^\circ) = \begin{bmatrix}
	\cos(45^\circ) & -\sin(45^\circ) \\
	\sin(45^\circ) & \cos(45^\circ)
\end{bmatrix}
$$

So we try:
\begin{align*}
	\mathbf{R}(45^\circ) \mathbf{R}_x &= \begin{bmatrix}
		\cos(45^\circ) & -\sin(45^\circ) \\
		\sin(45^\circ) & \cos(45^\circ)
	\end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \\
	&= \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
	\end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \\
	\mathbf{R}(45^\circ) \mathbf{R}_x & = \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} \\
		\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}}
	\end{bmatrix}
\end{align*}

To see if this worked, let’s apply this matrix to a vector pointing in the $ \widehat{i} $ direction (i.e., along the $ x $-axis):

$$
\big(\mathbf{R}(45^\circ) \mathbf{R}_x\big)\widehat{i} =
\begin{bmatrix}
	\tfrac{1}{\sqrt{2}} \\
	\tfrac{1}{\sqrt{2}}
\end{bmatrix}
$$

This result points at a 45$^\circ$ angle above the $ x $-axis—so far, so good. But remember, the goal is to reflect the vector across the $ y = x $ mirror. If our incoming ray is moving along $ \widehat{i} $, the reflected ray should emerge in the $ \widehat{j} $ direction. But that’s not what we’re getting!

<img
  src="{{ '/courses/math-methods/images/lec03/reflection1.png' | relative_url }}"
  alt="The image shows a horizontal red arrow labeled “Incoming Ray” pointing right toward a gray mirror line that is tilted upward to the right. At the point where the incoming ray meets the mirror, a second red arrow labeled “Outgoing Ray” reflects upward and to the right, leaving pointing directly along the mirror's surface."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

What went wrong?

It turns out that multiplying a matrix on the left by a rotation matrix only rotates the matrix’s *columns*—not the rows. Since matrices encode how vectors transform in both row and column space, this is only half the story.

To rotate both the rows and columns (i.e., the whole coordinate system), we must also apply the **transpose** of the rotation matrix on the **right**. So the full transformation becomes:

$$
\mathbf{R}_{y = x} = \mathbf{R}(45^\circ)\, \mathbf{R}_x\, \mathbf{R}(45^\circ)^T
$$

The matrix on the left rotates the column space, and the transpose on the right rotates the row space. Together, they fully rotate the transformation matrix. While this formula uses the transpose, the more general expression is $ \mathbf{R} \mathbf{A} \mathbf{R}^{-1} $, but because rotation matrices are orthogonal, their inverse is the same as their transpose. We’ll learn more about orthogonal matrices in Lecture 05.

Let’s compute it:
\begin{align*}
	\mathbf{R}_{y=x} &= \mathbf{R}(45^\circ) \mathbf{R}_x (\mathbf{R}(45^\circ))^\text{T} \\
	&= \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
	\end{bmatrix} \begin{bmatrix} 
		1 & 0 \\
		0 & -1 
	\end{bmatrix} \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} \\
		-\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
	\end{bmatrix} \\
	&= \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
	\end{bmatrix} \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} \\
		\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}}
	\end{bmatrix}  \\
	\mathbf{R}_{y=x} &= \begin{bmatrix}
		0 & 1 \\
		1 & 0
	\end{bmatrix}
\end{align*}

Now apply this to the vector $ \widehat{i} $:
$$
\mathbf{R}_{y = x} \widehat{i} =
\begin{bmatrix}
	0 & 1 \\
	1 & 0
\end{bmatrix}
\begin{bmatrix}
	1 \\ 0
\end{bmatrix}
=
\begin{bmatrix}
	0 \\ 1
\end{bmatrix}
$$

which correctly reflects the input vector along the line $ y = x $.

<img
  src="{{ '/courses/math-methods/images/lec03/reflection2.png' | relative_url }}"
  alt="The image shows a horizontal red arrow labeled “Incoming Ray” pointing right toward a gray mirror line that is tilted upward to the right. At the point where the incoming ray meets the mirror, a second red arrow labeled “Outgoing Ray” reflects upwards, forming an angle with the mirror’s surface."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

{% capture ex %}
To transform a **vector**, you only apply the transformation matrix to the column:
$$
\vec{r}\,' = \mathbf{R} \vec{r}
$$
where $ \mathbf{R} $ is the transformation from the current to the new coordinate system. Please note, some textbooks reverse the order and write transformations in terms of converting from the new to the old basis. Always check your convention!
{% endcapture %}
{% include result.html content=ex %}


{% capture ex %}
To transform a **matrix**, you must rotate both row and column space:

$$
\mathbf{A}' = \mathbf{R} \mathbf{A} \mathbf{R}^\text{T}
$$

This only works because $ \mathbf{R} $ is an orthogonal matrix. In general, for arbitrary transformations, we must write:

$$
\mathbf{A}' = \mathbf{R} \mathbf{A} \mathbf{R}^{-1}
$$
{% endcapture %}
{% include result.html content=ex %}








## Application: Light Ray Directions and Matrices

In geometric optics, light rays are represented by direction vectors, which indicate the path the light travels through space. When a light ray passes through certain optical components—such as rotating elements or mirrors—its direction changes. We can model these changes using rotation and reflection matrices.

Suppose we have a light ray traveling at a 45° angle above the $ x $-axis. This direction vector can be written as:

$$
\vec{v} = \begin{bmatrix} \cos 45^\circ \\ \sin 45^\circ \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}
$$

Now, let’s suppose this light ray:

1)  First passes through a device that rotates its direction counterclockwise by 30° (this could model, for example, a prism or a crystal of some kind),
2) Then reflects off a flat mirror that lies along the line $ y = x $.

We wish to find the final direction of the light ray after these two transformations.

### Step 1: Rotation by 30$^\circ$

The rotation matrix for a counterclockwise rotation by 30° is:

$$
\mathbf{R}(30^\circ) = \begin{bmatrix} \cos 30^\circ & -\sin 30^\circ \\ \sin 30^\circ & \cos 30^\circ \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix}
$$

Applying this to the original direction vector:


$$
\vec{v}_{\text{rotated}} = \mathbf{R}(30^\circ) \vec{v} = \begin{bmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3} - 1}{2\sqrt{2}} \\ \frac{\sqrt{3} + 1}{2\sqrt{2}} \end{bmatrix}
$$

So after rotating by 30°, the light ray is now traveling in a slightly steeper direction, more in the $y$-direction than in the $x$-direction.

### Step 2: Reflection Across the Line $ y = x $

To reflect a vector across the line $ y = x $, we use the reflection matrix:

$$
\mathbf{R}_{y=x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

This swaps the $ x $- and $ y $-components of the vector. So we apply:


$$
\vec{v}_{\text{final}} = \mathbf{R}_{y=x} \vec{v}_{\text{rotated}} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{\sqrt{3} - 1}{2\sqrt{2}} \\ \frac{\sqrt{3} + 1}{2\sqrt{2}} \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3} + 1}{2\sqrt{2}} \\ \frac{\sqrt{3} - 1}{2\sqrt{2}} \end{bmatrix}
$$

This new vector now represents the final direction of the light ray after both the rotation and the reflection. It has been bent by 30° and then mirrored across the 45° line.

### Physical Meaning

This example demonstrates how we can use matrices to simulate the physical transformations of light rays in an optical system. Rather than tracing rays with protractors and geometry, we can model these directional changes using algebra and matrix multiplication. This approach becomes especially powerful when modeling more complex systems, like multiple mirrors or lenses in a laser cavity or microscope.


{% capture ex %}
By chaining transformation matrices, you can model how the path of a light ray changes as it encounters mirrors, lenses, and other components. Reflections across axes or lines like $ y = x $ are represented by reflection matrices, while rotations are captured using rotation matrices. Order matters!
{% endcapture %}
{% include result.html content=ex %}











## Problem:

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


A charged particle is moving in the $xy$-plane with an initial velocity directed at a 30$^\circ$ angle from the $x$-axis. We can represent this velocity as a column matrix in the following manner:

$$
\vec{v} = \begin{bmatrix} v \cos 30^\circ \\ v \sin 30^\circ \end{bmatrix} = \begin{bmatrix} \frac{\sqrt{3}}{2} v \\ \frac{1}{2} v \end{bmatrix}
$$

where $v$ is the magnitude of the velocity.

Let's suppose the charged particle:

1) First encounters a magnetic field that rotates the velocity by 45$^\circ$ counterclockwise.
2) Then, it collides with a perfectly reflecting surface oriented along the line $y = -x$.


**Questions:**

a) Calculate the rotated velocity vector after the particle passes through the magnetic field.  
	
b) Apply the reflection matrix for the $ y = -x $ surface to find the velocity of the particle after the collision.  


> Hint: Carefully consider how a surface oriented along $ y = -x $ will affect the direction of a particle
>  reflecting off it, and how this can be represented with a matrix. You do not need to perform the same 
> transformations as above unless you wish to; you can derive the reflection matrix conceptually.