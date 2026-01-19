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

{% capture ex %}
To transform a **vector**, you apply the transformation matrix on the left:

$$
\vec{r}' = \mathbf{T} \vec{r}
$$

where $ \mathbf{T} $ transforms the original coordinates into the new ones. 

Note: Some textbooks use the reverse convention, so always pay careful attention to how a transformation is being defined.
{% endcapture %}
{% include result.html content=ex %}



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

which makes sense! 

Why? Because the vector $ (1,1) $ is already oriented at degrees above the positive $ x $-axis. Rotating it by an additional 45 degrees brings it all the way to the $ y $-axis, making the $ x $-component zero, which is it in our answer. Furthermore, the length of the original vector is $ \lvert \vec{r} \rvert = \sqrt{(1)^2 + (1)^2} = \sqrt{2} $, which agrees with the length of the new vector after simplifying $ \tfrac{2}{\sqrt{2}} = \sqrt{2}  $.
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

The rotation matrices for each of these rotations, as well as the initial vector, can be writted as:

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
\vec{r}' = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

In this case, **the order of operations matters**. To demonstrate this, we can try performing the rotations in the reverse order: first rotating about the $ x $-axis and then about the $ z $-axis. Starting with the initial vector:

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

Now, apply the rotation about the $ z $-axis:

$$
\mathbf{R}_z(90^\circ) \left( \mathbf{R}_x(90^\circ) \vec{r} \right) = \begin{bmatrix}
	0 & -1 & 0 \\
	1 & 0 & 0 \\
	0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
	1 \\ 0 \\ 0
\end{bmatrix} = \begin{bmatrix}
	0 \\ 1 \\ 0
\end{bmatrix} = \vec{r}''
$$

This result is different from our previous result ($\vec{r}' \ne \vec{r}''$), showing that the order of rotations affects the outcome. In mathematics, this property is called **non-commutativity**, meaning the sequence in which the operations, in this case rotations, are applied is important for getting the correct final result.
{% endcapture %}
{% include example.html content=ex %}

The fact that operations can be represented by matrices, and that the order of matrix multiplication often matters, is crucial for understanding Quantum Mechanics. Heisenberg was among the first to recognize that the order in which the position and momentum operators act on a quantum state matters. Specifically, he discovered that position and momentum do **not** commute. This non-commutativity, expressed in the canonical commutation relation

$$
[\widehat{x}, \widehat{p}_x] = \widehat{x} \widehat{p}_x - \widehat{p}_x \widehat{x} = i \hbar
$$

was a key insight that led Heisenberg, with contributions from Born and Jordan, to develop *Matrix Mechanics*, one of the earliest formulations of Quantum Mechanics.

At the time, matrices were relatively new to physics, and it was Jordan who introduced Heisenberg to matrices and helped guide the a good portion of the mathematical formulation. This collaboration laid the foundation for much of modern quantum theory.







### Properties of Rotation Matrices

Previously we claimed that the determinant of a rotation matrix is always $+1$. This tells us that no combination of rotation matrices can ever create a reflection (which would have determinant $-1$ as we’ll soon see). Let’s confirm this for the $2 \times 2$ rotation matrix we derived earlier:

$$
\begin{aligned}
\text{det}\big(\mathbf{R}_z(\theta)\big) &= 
\begin{vmatrix}
	\cos(\theta) & -\sin(\theta) \\
	\sin(\theta) & \cos(\theta)
\end{vmatrix} \\
&= \cos^2(\theta) - (-\sin^2(\theta)) \\ 
&= \cos^2(\theta) + \sin^2(\theta) \\
 \text{det}\big(\mathbf{R}_z(\theta)\big) & = 1
\end{aligned}
$$

Formally, this property means rotation matrices are part of the **special orthogonal group**, denoted $SO(n)$, where $n$ is the dimension of the space (e.g., $SO(2)$ for 2D rotations, $SO(3)$ for 3D rotations).

The term *special* means that the determinant is $+1$, and we’ll get into what *orthogonal* means in Lecture 05. For now, the key takeaway is that rotations preserve the “handedness” (or orientation) of the coordinate system. Further, they do not change the length of the vectors they transform.

These special orthogonal groups are incredibly important in physics, especially in particle physics and quantum field theory, where rotations represent fundamental symmetries of nature.

Another important property of rotation matrices, in that **successive rotations** can be represented by multiplying rotation matrices together. If $\mathbf{R}_1(\theta)$ and $\mathbf{R}_2(\phi)$ are rotation matrices, then their product is also a rotation matrix:

$$
\mathbf{R}(\theta,\phi) = \mathbf{R}_1(\theta) \mathbf{R}_2(\phi)
$$

This resulting matrix $\mathbf{R}$ represents the combined effect of applying $\mathbf{R}_2$ followed by $\mathbf{R}_1$ (remember: right to left!).

It's important to note remember, we cannot stress this enough, that **matrix multiplication is not commutative**. That means:

$$
\mathbf{R}_1 \mathbf{R}_2 \neq \mathbf{R}_2 \mathbf{R}_1
$$

This is especially true in 3D space or higher, where the order of applying rotations significantly affects the outcome.








## Reflections

Another interesting coordinate transformation made possible using matrices is **reflection**. Reflections occur when you mirror a vector across one (or more) axes. This effectively flips one of the coordinates, say $x$ becomes $-x$, while leaving the others unchanged.

In general, a reflection matrix transforms points across a specific line (in 2D) or plane (in 3D). For example, reflecting a point across the $x$-axis in two dimensions is done using:

$$
\mathbf{R}_{x} = \begin{bmatrix}
	1 & 0 \\
	0 & -1
\end{bmatrix}
$$

If we apply this matrix to the vector $(2, 5)$:

$$
\begin{bmatrix}
	1 & 0 \\
	0 & -1
\end{bmatrix}
\begin{bmatrix}
	2 \\ 5
\end{bmatrix}
= 
\begin{bmatrix}
	2 \\ -5
\end{bmatrix}
$$

we see that the $y$-value is flipped while the $x$-value remains unchanged.

In three dimensions, reflecting a point across the $xy$-plane is represented by:

$$
\mathbf{R}_{xy} = \begin{bmatrix}
	1 & 0 & 0 \\
	0 & 1 & 0 \\
	0 & 0 & -1
\end{bmatrix}
$$

{% capture ex %}
Consider a scenario in geometric optics where a light ray strikes a flat mirror aligned with the $x$-axis in a 2D plane; maybe mounted to the ceiling for a dance party, who knows?

Suppose a light ray is initially traveling at a 45° angle above the $x$-axis. When it hits the mirror, its $y$-component flips sign, while its $x$-component stays the same, assuming $x$ is horizontal and $y$ is vertical. The ray goes from traveling up and to the right, to down and to the right—a perfect reflection across the $x$-axis.

To model this reflection mathematically, we use the reflection matrix:

$$
\mathbf{R}_{x} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

As we have seen, this flips the $y$-component and leaves the $x$-component untouched.

Suppose the incoming light ray is:

$$
\vec{r}_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

Then the reflected ray is:

$$
\vec{r}_{\text{reflected}} = \mathbf{R}_x \, \vec{r}_0 
= 
\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} 
\begin{bmatrix} 1 \\ 1 \end{bmatrix} 
= 
\begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

This new vector points 45° below the $x$-axis, just as we expect from a mirror reflection.

In optics, this simple model generalizes to more complex setups, like: angled mirrors, curved surfaces, and higher-dimensional systems—where reflection matrices help track how light (or particles) change direction.
{% endcapture %}
{% include example.html content=ex %}

### Properties of Reflection Matrices

A key feature of reflection matrices is that they have a determinant of $-1$ (go ahead and verify this for both reflection matrices we’ve seen so far). This tells us they **reverse the handedness** or orientation of the coordinate system.

This is a fundamental difference from rotation matrices, which always preserve orientation (having determinant $+1$). In particle physics, flipping the handedness of a system is called a **parity transformation**. Reflection matrices are crucial for studying systems where **parity is violated**, which occurs in certain subatomic processes. So, while reflections may seem like a geometric curiosity, they actually connect to some of the deepest symmetries—and asymmetries—of the physical world.








## Rescaling / Scaling

Rescaling (or scaling) matrices are used to change the size of geometric objects. These matrices stretch or compress vectors by multiplying them by a scaling factor in one or more directions. 

In two dimensions, a general rescaling matrix has the form:

$$
\mathbf{S} = \begin{bmatrix}
	s_x & 0 \\
	0 & s_y
\end{bmatrix}
$$

where $s_x$ and $s_y$ are the scaling factors along the $x$- and $y$-axes, respectively. In three dimensions, the scaling matrix becomes:

$$
\mathbf{S} = \begin{bmatrix}
	s_x & 0 & 0 \\
	0 & s_y & 0 \\
	0 & 0 & s_z
\end{bmatrix}
$$

where $s_z$ is the scaling factor in the $z$-direction.

### Properties of Rescaling / Scaling Matrices

Unlike rotation or reflection matrices, rescaling matrices can have **any** determinant—positive, negative, or even zero—depending on how they stretch or compress space:

- A **positive** determinant means the transformation preserves orientation.
- A **negative** determinant means the transformation includes a reflection, flipping orientation.
	- Simple reflection matrices are a special case of scaling matrices where one or more axes are multiplied by $-1$.
- A **zero** determinant collapses space entirely (e.g., flattening 3D space onto a plane), which means the transformation is **not invertible**.







## Transforming Matrices

At this point, you might be wondering: how do we reflect across axes other than the $x$- or $y$-axis?

And your instinct might be, “If I can rotate a mirror in real life, maybe I can just rotate the reflection matrix too!”

That intuition is spot on, but we haven’t yet talked about how to rotate a matrix itself. So let’s try to figure it out this can be done by experimenting with reflection and rotation matrices.

Suppose we want a reflection matrix for a mirror placed along the line $y = x$. That is, a mirror that lies 45$^\circ$ above the horizontal. One idea is to take the standard reflection across the $x$-axis, $ \mathbf{R}_x $, and rotate it by 45$^\circ$ using the rotation matrix:

$$
\mathbf{R}_z(45^\circ) = \begin{bmatrix}
	\cos(45^\circ) & -\sin(45^\circ) \\
	\sin(45^\circ) & \cos(45^\circ)
\end{bmatrix} = \begin{bmatrix}
	\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
	\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
\end{bmatrix}
$$

So we try:

$$
\begin{aligned}
	\mathbf{R}_z(45^\circ) \mathbf{R}_x &= \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
	\end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \\
	&= \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} \\
		\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}}
	\end{bmatrix}
\end{aligned}
$$

To check if this worked properly, let’s apply this matrix to the unit vector $ \widehat{i} $ (pointing along the $x$-axis):

$$
\left(\mathbf{R}(45^\circ) \mathbf{R}_x\right)\widehat{i} =
\begin{bmatrix}
	\tfrac{1}{\sqrt{2}} \\
	\tfrac{1}{\sqrt{2}}
\end{bmatrix}
$$

This result points 45$^\circ$ above the $x$-axis. This isn't correct. Remember, we’re trying to reflect across the $y = x$ line. If our incoming ray is along $ \widehat{i} $, we expect the reflected ray to point along $ \widehat{j} $. But that’s not what we’re getting. What we are getting it pictured here:

<img
  src="{{ '/courses/math-methods/images/lec03/reflection1.png' | relative_url }}"
  alt="The image shows a horizontal red arrow labeled “Incoming Ray” pointing right toward a gray mirror line that is tilted upward to the right. At the point where the incoming ray meets the mirror, a second red arrow labeled “Outgoing Ray” reflects upward and to the right, leaving pointing directly along the mirror's surface."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

So what went wrong?

It turns out that multiplying a matrix on the **left** by a rotation matrix only rotates the matrix’s **columns**—not its rows. Since matrices encode transformations in both row and column space, this only gives us part of the picture.

To rotate both rows *and* columns—that is, to rotate the entire coordinate system—we also need to apply the **transpose** of the rotation matrix on the **right**. The full transformation becomes:

$$
\mathbf{R}_{y = x} = \mathbf{R}(45^\circ)\, \mathbf{R}_x\, \mathbf{R}(45^\circ)^T
$$

The matrix on the left rotates the columns, and the transpose on the right rotates the rows. Together, they generate the whole matrix transformation. 

{% capture ex %}
While this matrix transoformation derivation ended up using a transpose on the right, the more general formula is:

$$
\mathbf{R} \mathbf{A} \mathbf{R}^{-1}
$$

Since rotation matrices are orthogonal, their inverse equals their transpose. (We’ll cover this in more detail in Lecture 05.)
{% endcapture %}
{% include warning.html content=ex %}


Let’s compute it:

$$
\begin{aligned}
	\mathbf{R}_{y=x} &= \mathbf{R}(45^\circ) \mathbf{R}_x \mathbf{R}(45^\circ)^T \\
	&= \begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} \\
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
	\end{bmatrix} 
	\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} 
	\begin{bmatrix}
		\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} \\
		-\tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}
	\end{bmatrix} \\
	&= \begin{bmatrix}
		0 & 1 \\
		1 & 0
	\end{bmatrix}
\end{aligned}
$$

Now let’s apply this to $ \widehat{i} $ :  

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

which is exactly what we expect—a reflection of $ \widehat{i} $ across the line $y = x$.

<img
  src="{{ '/courses/math-methods/images/lec03/reflection2.png' | relative_url }}"
  alt="The image shows a horizontal red arrow labeled “Incoming Ray” pointing right toward a gray mirror line that is tilted upward to the right. At the point where the incoming ray meets the mirror, a second red arrow labeled “Outgoing Ray” reflects upwards, forming an angle with the mirror’s surface."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

{% capture ex %}
To transform a **matrix**, you must rotate both its row and column space:

$$
\mathbf{A}' = \mathbf{R} \mathbf{A} \mathbf{R}^T
$$

This works when $ \mathbf{R} $ is an orthogonal matrix. In general, the transformation is:

$$
\mathbf{A}' = \mathbf{R} \mathbf{A} \mathbf{R}^{-1}
$$
{% endcapture %}
{% include result.html content=ex %}








## Application: Light Ray Directions and Matrices

In geometric optics, light rays are often represented by direction vectors, which describe the path light takes through space. When a light ray passes through optical components, like rotating elements or mirrors, its direction changes. We can model these changes using rotation and reflection matrices.

Suppose we have a light ray traveling at a 45° angle above the $x$-axis. This direction vector can be written as:

$$
\vec{v} = \begin{bmatrix} \cos 45^\circ \\ \sin 45^\circ \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}
$$

Now imagine this light ray:

1. First passes through a device that rotates its direction counterclockwise by 30° (this could model, for example, a prism or a birefringent crystal),  
2. Then reflects off a flat mirror that lies along the line $y = x$.

We want to determine the final direction of the light ray after these two transformations.

### Step 1: Rotation by 30$^\circ$

The rotation matrix for a counterclockwise rotation by 30° is:

$$
\mathbf{R}(30^\circ) = \begin{bmatrix} \cos 30^\circ & -\sin 30^\circ \\ \sin 30^\circ & \cos 30^\circ \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix}
$$

Applying this to the original direction vector:

$$
\vec{v}_{\text{rotated}} = \mathbf{R}(30^\circ) \vec{v} 
= \begin{bmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix}
\begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3} - 1}{2\sqrt{2}} \\ \frac{\sqrt{3} + 1}{2\sqrt{2}} \end{bmatrix}
$$

So after rotating by 30°, the light ray is now directed slightly more steeply—more vertical than horizontal.

### Step 2: Reflection Across the Line $y = x$

To reflect a vector across the line $y = x$, we use the reflection matrix:

$$
\mathbf{R}_{y=x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

This matrix simply swaps the $x$- and $y$-components. So we apply:

$$
\vec{v}_{\text{final}} = \mathbf{R}_{y=x} \vec{v}_{\text{rotated}} = 
\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} 
\begin{bmatrix} \frac{\sqrt{3} - 1}{2\sqrt{2}} \\ \frac{\sqrt{3} + 1}{2\sqrt{2}} \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3} + 1}{2\sqrt{2}} \\ \frac{\sqrt{3} - 1}{2\sqrt{2}} \end{bmatrix}
$$

This final vector represents the direction of the light ray after undergoing both the rotation and reflection. It has been rotated and then reflected across the 45° line, changing both its orientation and direction.

### Physical Meaning

This example shows how matrices let us simulate the behavior of light rays as they interact with mirrors, prisms, or other optical elements. Instead of drawing rays and measuring angles by hand, we can calculate their transformations using matrix algebra. This method becomes especially useful when analyzing more complex systems—like multiple reflections inside a laser cavity, or light passing through layers of lenses in a microscope.

{% capture ex %}
By chaining transformation matrices, you can model how the path of a light ray changes as it encounters mirrors, lenses, and other components. Reflections across axes or lines like $y = x$ are represented by reflection matrices, while rotations are captured using rotation matrices. Order matters!
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
\mathbf{R}_z(30^\circ) = \begin{bmatrix} \cos(30^\circ) & -\sin(30^\circ) \\ \sin(30^\circ) & \cos(30^\circ) \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix}
$$

Applying this to the original direction vector:

$$
\vec{v}_{\text{rotated}} = \mathbf{R}_z(30^\circ) \vec{v} = \begin{bmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3} - 1}{2\sqrt{2}} \\ \frac{\sqrt{3} + 1}{2\sqrt{2}} \end{bmatrix}
$$

After rotating by 30°, the light ray is now traveling in a slightly steeper direction, more in the $y$-direction than in the $x$-direction.

### Step 2: Reflection Across the Line $ y = x $

To reflect a vector across the line $ y = x $, we use the reflection matrix we found previously:

$$
\mathbf{R}_{y=x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

This swaps the $ x $- and $ y $-components of the vector. So we apply:


$$
\vec{v}_{\text{final}} = \mathbf{R}_{y=x} \vec{v}_{\text{rotated}} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{\sqrt{3} - 1}{2\sqrt{2}} \\ \frac{\sqrt{3} + 1}{2\sqrt{2}} \end{bmatrix} 
= \begin{bmatrix} \frac{\sqrt{3} + 1}{2\sqrt{2}} \\ \frac{\sqrt{3} - 1}{2\sqrt{2}} \end{bmatrix}
$$

This new vector now represents the final direction of the light ray after both the rotation and the reflection. It has been rotated by 30° and then mirrored across the 45° line.

### Physical Meaning

This example demonstrates how we can use matrices to simulate the physical transformations of light rays in an optical system. Rather than tracing rays with protractors and geometry, we can model these directional changes using algebra and matrix multiplication. This approach becomes especially powerful when modeling more complex systems, like multiple mirrors or lenses in a laser cavity or microscope.

{% capture ex %}
By chaining transformation matrices, you can model how the path of a light ray changes as it encounters mirrors, lenses, and other components. Reflections across axes or lines, like $ y = x $, are represented by reflection matrices, while rotations are captured using rotation matrices. Order matters!
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