---
layout: default
title: Mathematical Methods - Lecture 04
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 4
---


# Lecture 04 – Vector Operations: Cross and Dot Products

Let’s take a break from matrix operations and return to vector operations. It might seem a bit odd that we started matrices before finishing our discussion of vectors, but there’s a reason for that: understanding how to compute the determinant of a $3 \times 3$ matrix will come in handy when we go to calculate some vector operations.

In this lecture, we’ll focus on two key operations involving vectors:

- The **dot product** (also called the **scalar product**)
- The **cross product** (also called the **vector product**)

To reinforce the idea that matrices can be thought of as **operators**, mathematical objects that represent an operation of some kind, we’ll express these operations in matrix form when it’s meaningful and helpful to do so.


## Dot (Scalar) Product

The dot product, or scalar product, returns a **scalar quantity**. It’s widely used in physics, for example in the calculation of work done by a constant force:

$$
W = \vec{F} \cdot \Delta \vec{x}
$$

where $W$ is the work done by the force $\vec{F}$ applied over the displacement $\Delta \vec{x}$. This formulation, as you may remember, ony works if the force is constant in magnitude and direction.

If the force is not constant, we must calculate an infinitesimal amount of work done by the force over an infinitesimally small displacement $d\vec{x}$ and then add up all of these tiny contributions to get the whole. That is, we must integrate over the entire path:

$$
W = \int\limits_\text{Path} \vec{F} \cdot d\vec{x}
$$

While this application is common, it doesn’t quite capture the full versatility of the dot product.

### Geometric Interpretation of the Dot Product

A more general and insightful interpretation is that the dot product measures **how much one vector points in the direction of another**. Formally:

$$
\vec{A} \cdot \vec{B} = |\vec{A}| |\vec{B}| \cos(\theta_{AB}) = AB \cos(\theta_{AB})
$$

where:
- $A$ and $B$ are the magnitudes of vectors $\vec{A}$ and $\vec{B}$,
- $\theta_{AB}$ is the smallest angle between them when placed tail-to-tail.

This definition shows that the dot product gets larger as the two vectors point more in the same direction—and becomes zero when the vectors are perpendicular.

<img
  src="{{ '/courses/math-methods/images/lec04/Components.png' | relative_url }}"
  alt="A diagram showing vector A drawn in blue along a slanted horizontal direction. Vector B, drawn in red, extends upward and to the left forming an angle theta A B with vector A. A dashed black line shows the projection of B onto A, labeled B parallel to A. A gray dashed vertical segment from the tip of B down to the projection line represents the perpendicular component, labeled A perpendicular to B. The angle theta A B between vectors A and B is marked near their tails."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

In the figure above, the component of $\vec{B}$ **perpendicular** to $\vec{A}$ is denoted by $B_{\perp A}$. Using basic trigonometry, we can write:

$$
B_{\perp A} = B \sin(\theta_{AB})
$$

Similarly, the component of $\vec{B}$ **parallel** to $\vec{A}$ is:

$$
B_{\parallel A} = B \cos(\theta_{AB})
$$

If we plug this into the definition of the dot product from earlier, we get:

$$
\vec{A} \cdot \vec{B} = A B \cos(\theta_{AB}) = A B_{\parallel A}
$$

This gives an intuitive interpretation: the dot product tells us **how much of $\vec{B}$ points in the direction of $\vec{A}$**, scaled by the length of $\vec{A}$.

You're welcome to show, using a similar diagram, that:

$$
A_{\perp B} = A \sin(\theta_{AB}) \qquad \text{and} \qquad A_{\parallel B} = A \cos(\theta_{AB})
$$

The second of these leads to an alternate expression for the dot product:  

$$
\vec{A} \cdot \vec{B} = A B_{\parallel A} \qquad \text{or} \qquad \vec{A} \cdot \vec{B} = A_{\parallel B} B
$$

Notice that it doesn’t actually matter **which vector is projected onto the other**, the result is the same either way. Here, **projected** is a mathematical term meaning to draw the component of one vector that lies parallel, along another. We’ll explore the **projection operator** more formally in a future lecture.

### Properties of the Dot Product

The geometric interpretation of the dot product gives rise to several useful properties and applications. Here are some of the general properties you are welcome to workthrough and prove to yourself, if we haven't done so already:

1. **Orthogonality**: If two vectors are orthogonal (i.e., perpendicular), their dot product is zero:

   $$
   \vec{A} \cdot \vec{B} = 0 \quad \text{if} \quad \theta_{AB} = 90^\circ
   $$

   The only way for one vector to have zero component parallel to another is if the two are perpendicular.

2. **Magnitude of a Vector**: The dot product of a vector with itself gives the square of its magnitude:

   $$
   \vec{A} \cdot \vec{A} = A^2
   $$

   Thsi is an important notation note worth being more direct about. When we write the magnitude squared of a vector like $A^2$ what we really mean is the vector dotted with itself. This means the magnitude of a vector $ \vec{A} $ is more formally and technically written as:

   $$
   |\vec{A}| = A = \sqrt{\vec{A} \cdot \vec{A}}
   $$

3. **Commutativity**: The dot product is commutative:

   $$
   \vec{A} \cdot \vec{B} = \vec{B} \cdot \vec{A}
   $$

   This holds because both expressions equal $AB\cos(\theta_{AB})$. Switching the order of the vectors doesn’t change the angle between them.

4. **Associativity with Scalars**: The dot product behaves nicely with scalar multiplication:

   $$
   (c \vec{A}) \cdot \vec{B} = c (\vec{A} \cdot \vec{B})
   $$

   where $c$ is a scalar. This follows from:

   $$
   |c\vec{A}|\,|\vec{B}|\,\cos(\theta_{AB}) = c \left(|\vec{A}|\,|\vec{B}|\,\cos(\theta_{AB})\right)
   $$

   - For **positive** $c$, the angle $\theta_{AB}$ is unchanged:

   $$
   c |\vec{A}|\,|\vec{B}|\,\cos(\theta_{AB}) = c |\vec{A}|\,|\vec{B}|\,\cos(\theta_{AB})
   $$

   - For **negative** $c$, the angle shifts to $180^\circ - \theta_{AB}$ (multiplying a vector by a negative flips its direction; draw a sketch to see how this changes the angle between the vectors), and we can write:

   $$
   |c|\,|\vec{A}|\,|\vec{B}|\,\cos(180^\circ - \theta_{AB}) = -|c|\,|\vec{A}|\,|\vec{B}|\,\cos(\theta_{AB}) = c |\vec{A}|\,|\vec{B}|\,\cos(\theta_{AB})
   $$

   where $c = -\lvert c\rvert$ so that $c$ is negative.

   So the property holds regardless of the sign of $c$.

5. **Distributivity**: The dot product distributes over vector addition:

   $$
   \vec{A} \cdot (\vec{B} + \vec{C}) = \vec{A} \cdot \vec{B} + \vec{A} \cdot \vec{C}
   $$

   This is especially helpful for expanding dot products in component form or simplifying expressions in physics.









The last property can be challenging to see with the geometric definition of the dot product alone. To gain more insight, let’s introduce a component-based definition of the dot product.

Consider a vector $ \vec{A} $ with components along the $ x $-, $ y $-, and $ z $-axes, which we’ll denote as $ A_x $, $ A_y $, and $ A_z $, respectively. If we imagine vectors of unit length (magnitude of 1) pointing along each axis, we can express $ \vec{A} $ in terms of its projections on each axis. Conveniently, we already have these unit vectors, denoted $ \hat{i} $, $ \hat{j} $, and $ \hat{k} $ for the $ x $-, $ y $-, and $ z $-axes, respectively. This allows us to write:

$$
\vec{A} = A_x \,\hat{i} + A_y \,\hat{j} + A_z \,\hat{k}
$$

where the unit vectors for each axis obey the following properties -- coming directly from our previous discussion of the dot product:

$$ \hat{i} \cdot \hat{i} = 1 \qquad  \hat{j} \cdot \hat{j} = 1 \qquad\hat{k} \cdot \hat{k} = 1 \quad \text{and} \quad \hat{i} \cdot \hat{j} = 0 \qquad  \hat{i} \cdot \hat{k} = 0 \qquad\hat{j} \cdot \hat{k} = 0 $$

From these definitions we can investigate what the dot product of two vectors looks like in component form:

$$ \vec{A} \cdot \vec{B} = \left( A_x \,\hat{i} + A_y \,\hat{j} + A_z \,\hat{k} \right) \cdot \left( B_x \,\hat{i} + B_y \,\hat{j} + B_z \,\hat{k} \right) $$

Expanding (or the much more enjoyable word ``FOILing") gives:


$$ \begin{aligned}
	\vec{A} \cdot \vec{B} &= A_x B_x \,\hat{i} \cdot \hat{i} + A_x B_y \,\hat{i} \cdot \hat{j} + A_x B_z \,\hat{i} \cdot \hat{k} \\
	&+ A_y B_x \,\hat{j} \cdot \hat{i} + A_y B_y \,\hat{j} \cdot \hat{j} + A_y B_z \,\hat{j} \cdot \hat{k} \\
	 &+ A_z B_x \,\hat{k} \cdot \hat{i} + A_z B_y \,\hat{k} \cdot \hat{j} + A_z B_z \,\hat{k} \cdot \hat{k}\\[0.75ex]
	 \vec{A} \cdot \vec{B} &= A_x B_x \,\hat{i} \cdot \hat{i} + A_x B_y \,(0) + A_x B_z \,(0)\\
	 &+ A_y B_x \,(0) + A_y B_y \,\hat{j} \cdot \hat{j} + A_y B_z \,(0) \\
	 &+ A_z B_x \,(0) + A_z B_y \,(0) + A_z B_z \,\hat{k} \cdot \hat{k}\\[0.75ex]
	 \vec{A} \cdot \vec{B} &= A_x B_x \,\hat{i} \cdot \hat{i} +  A_y B_y \,\hat{j} \cdot \hat{j} + A_z B_z \,\hat{k} \cdot \hat{k} \\[0.75ex]
	 \vec{A} \cdot \vec{B} &= A_x B_x +  A_y B_y + A_z B_z
\end{aligned}   $$

As a result of the definitions of the unit vectors and their dot products, the dot product between two vectors in components is given as:
$$ \vec{A} \cdot \vec{B} = A_x B_x +  A_y B_y + A_z B_z $$

This representation of the dot product makes the verification of the last property listed previously much easier -- you are welcome to check it on your own. 



{% capture ex %}
Suppose you have an object being acted on by multiple forces. You want to calculate the work done to the object by these forces. The forces are given as:
$$ \vec{F}_1 = 10\,\text{N} \,\hat{i} - 30\,\text{N} \,\hat{j} \qquad \text{and} \qquad \vec{F}_2 = -15\,\text{N} \,\hat{i} - 20\,\text{N} \,\hat{j} $$

The displacement of the object is given as:
$$ \Delta \vec{x} = 5\,\text{m} \,\hat{i} - 3\,\text{m} \,\hat{j}   $$

The work done by the first force can be found to be:
$$ W_1 = \vec{F}_1 \cdot \Delta \vec{x} = \left(10\,\text{N} \,\hat{i} - 30\,\text{N}\right) \cdot \left( 5\,\text{m} \,\hat{i} - 3\,\text{m} \,\hat{j} \right) =  50\,\text{J} \, \left( \hat{i} \cdot \hat{i}\right) + 90\,\text{J} \, \left( \hat{j} \cdot \hat{j}\right) = 140\,\text{J}  $$

The work done by the second force can be found to be:
$$ W_2 = \vec{F}_2 \cdot \Delta \vec{x} = \left(-15\,\text{N} \,\hat{i} - 20\,\text{N} \,\hat{j}\right) \cdot \left( 5\,\text{m} \,\hat{i} - 3\,\text{m} \,\hat{j} \right) =  -75\,\text{J} \, \left( \hat{i} \cdot \hat{i}\right) + 60\,\text{J} \, \left( \hat{j} \cdot \hat{j}\right) = -15\,\text{J}  $$

The total work comes out to be:
$$ W_{tot} = W_1 + W_2 = 125\,\text{J} $$

Alternatively, we could have found the total force first:
$$ \vec{F}_{tot} = \vec{F}_1 + \vec{F}_2 = \left( 10\,\text{N} \,\hat{i} - 30\,\text{N} \,\hat{j}\right) + \left( -15\,\text{N} \,\hat{i} - 20\,\text{N} \,\hat{j}\right) = -5\,\text{N} \,\hat{i} - 50\,\text{N} \,\hat{j} $$
and then found the total work using that:
$$ W_{tot} = \vec{F}_{tot} \cdot \Delta \vec{x} = \left( -5\,\text{N} \,\hat{i} - 50\,\text{N} \,\hat{j} \right) \cdot \left( 5\,\text{m} \,\hat{i} - 3\,\text{m} \,\hat{j} \right) =  -25\,\text{J} \, \left( \hat{i} \cdot \hat{i}\right) + 150\,\text{J} \, \left( \hat{j} \cdot \hat{j}\right) = 125\,\text{J}    $$
which is the same as we got previously.
{% endcapture %}
{% include example.html content=ex %}











## Cross (Vector) Product

The cross product, also called the vector product because it yields a vector result, is another vector operation frequently used in physics. For example, it appears in the calculation of the angular momentum of a particle:

$$
\vec{L} = \vec{r} \times \vec{P}
$$

and in the expression for the magnetic force on a moving charged particle:

$$
\vec{F}^B = q(\vec{v} \times \vec{B}).
$$

Let's discuss the cross product similarly to our treatment of the dot product by first examining its geometric interpretation. The geometric interpretation of the cross product is that it measures the perpendicular component of one vector scaled by the length of the second vector. Mathematically, we can define the magnitude of the cross product of two vectors as:


$$ \lvert \vec{A} \times \vec{B} \rvert = \lvert \vec{A}\rvert \lvert\vec{B}\rvert \sin(\theta_{AB}) = A B \sin(\theta_{AB}) $$

where $ A $ and $ B $ are the magnitudes of vectors $ \vec{A} $ and $ \vec{B} $, respectively, $ \theta_{AB} $ is the smallest angle between the two vectors when drawn tail-to-tail.

Remember when we were looking at the vector diagram given in the discussion of the dot product where we found that the component of $ \vec{B} $ perpendicular to $ \vec{A} $ is given by:

$$ B_{\perp A} = B \sin(\theta_{AB}) $$

Using this lets us rewrite the magnitude of the cross product as:

$$ \lvert\vec{A} \times \vec{B}\rvert = A B \sin(\theta_{AB}) = A B_{\perp A} $$

You could also show that $A_{\perp B} = A \sin(\theta_{AB})$ using a similar vector diagram -- which gives:

$$ \lvert\vec{A} \times \vec{B}\rvert = A B_{\perp A} = A_{\perp B} B $$

So it ultimately doesn't matter which vector is considered as having the perpendicular component; the calculation yields the same magnitude.

Now for the question, **``how do we get the direction?"**--we did say the result should be a vector after all. This is where something called the **right-hand-rule (RHR)** comes into play. The resulting vector from a cross product will:

- point perpendicular to both of the original vectors involved in the cross product, and
- point in the direction given by the RHR. 


Here is how the RHR works for $\vec{A}\times\vec{B}$:

1. Open your **right hand** to look like an oven mitt -- the kind with a place for your thumb and a place for all of your other fingers. Point the fingers of your right hand the direction of the first vector, $ \vec{A} $.
	
2. Wrap your fingers, not your thumb, so that they naturally curl towards the second vector, $ \vec{B} $, in the direction angle $\theta_{AB}$ is measured. 
	
3. Your thumb, extended perpendicular to your fingers, points in the direction of the cross product, $ \vec{A} \times \vec{B} $.

Notice, this rule implies the vector created from taking $ \vec{A} \times \vec{B} $ will point in the opposite direction as the vector that comes from $ \vec{B} \times \vec{A} $. Check this out for yourself! 

The cross product $ \vec{A} \times \vec{B} $ has several useful properties, similar to the ones for the dot product:


1. **Zero Cross Product**: Two vectors have a cross product of zero if they are parallel or anti-parallel-- point in the same or opposite direction -- as there is no perpendicular component:

	$$
	\vec{A} \times \vec{B} = \vec{0} \quad \text{if} \quad \vec{A} \parallel \vec{B} 
	$$
	
2. **Orthogonality of Result**: The resulting vector $ \vec{A} \times \vec{B} $ is perpendicular to both $ \vec{A} $ and $ \vec{B} $, aligning with the right-hand rule. This means if you try to take $\vec{A}$ or $\vec{B}$ and take the dot product with $ \vec{A} \times \vec{B} $ you will get zero, always!

	$$ \vec{A} \cdot (\vec{A} \times \vec{B}) = 0 \qquad \vec{B} \cdot (\vec{A} \times \vec{B}) = 0 $$
	
3. **Magnitude of the Cross Product**: The magnitude of $ \vec{A} \times \vec{B} $ represents the area of the parallelogram formed by $ \vec{A} $ and $ \vec{B} $ (we will talk about this in a bit):

	$$
	\lvert\vec{A} \times \vec{B}\rvert = \lvert\vec{A}\rvert \lvert\vec{B}\rvert \sin(\theta_{AB})
	$$

	where, again, $ \theta_{AB} $ is the angle between $ \vec{A} $ and $ \vec{B} $ when draw tail-to-tail. This area interpretation is useful in physics when dealing with planar areas, but is only really useful in specific situations.
	
4.  **Anti-commutative Property**: The cross product is anti-commutative, meaning:

	$$
	\vec{A} \times \vec{B} = -(\vec{B} \times \vec{A})
	$$

	We saw this in our discussion of the RHR just a little bit ago.
	
5. **Scalar Multiplication**: For any scalar $ c $, the cross product behaves as follows:

	$$
	(c \vec{A}) \times \vec{B} = c (\vec{A} \times \vec{B}) = \vec{A} \times (c \vec{B})
	$$

	You can prove this in a similar manner as we did with the dot product. You can work this out if you are interested.
	
6. **Distributive Property**: The cross product distributes over vector addition:

	$$
	\vec{A} \times (\vec{B} + \vec{C}) = (\vec{A} \times \vec{B}) + (\vec{A} \times \vec{C})
	$$


As with the dot product, the last property listed above can be difficult to see using only the geometric interpretation. To gain further insight, let’s consider the component-based interpretation of the cross product.

For a right-handed coordinate system, where the unit vectors $ \hat{i} $, $ \hat{j} $, and $ \hat{k} $ follow the cross product’s RHR, we have the following relationships:
$$
\hat{i} \times \hat{j} = \hat{k} \qquad \hat{j} \times \hat{k} = \hat{i} \qquad \hat{k} \times \hat{i} = \hat{j}
$$

The first relation, $ \hat{i} \times \hat{j} = \hat{k} $, is typically used to define a coordinate system as right-handed. 


<img
  src="{{ '/courses/math-methods/images/lec04/UnitVectors.png' | relative_url }}"
  alt="The image shows a three-dimensional coordinate system with black arrows marking the positive x, y, and z directions. The x-axis extends horizontally to the right, the y-axis extends vertically upward, and the z-axis extends diagonally downward and to the left. Three unit vectors are drawn in red: one pointing along the positive x direction, one pointing along the positive y direction, and one pointing along the negative z direction. Each axis is labeled with its corresponding letter, and each red arrow is labeled with the name of its unit vector."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">


This convention ensures consistency in calculations and aligns with the right-hand rule. The anti-commutation property of the cross product gives us the other 3 possible options for the unit vector cross products:

$$
\hat{j} \times \hat{i} = -\hat{k} \qquad \hat{k} \times \hat{j} = -\hat{i} \qquad \hat{i} \times \hat{k} = - \hat{j}
$$

because the unit vectors crossed with themselves will be zero:

$$
\hat{i} \times \hat{i} = 0  \qquad \hat{j} \times \hat{j} = 0 \qquad \hat{k} \times \hat{k} = 0
$$

From these results, we can take the cross product of two vectors in component for to get:


$$ \begin{aligned}
		\vec{A} \times \vec{B} &= A_x B_x \,\hat{i} \times \hat{i} + A_x B_y \,\hat{i} \times \hat{j} + A_x B_z \, \hat{i} \times \hat{k} \\[0.75ex]
		&+ A_y B_x \,\hat{j} \times \hat{i} + A_y B_y \, \hat{j} \times \hat{j} + A_y B_z \,\hat{j} \times \hat{k}\\[0.75ex]
		&+ A_z B_x \,\hat{k} \times  \hat{i} + A_z B_y \,\hat{k} \times  \hat{j} + A_z B_z \,\hat{k} \times \hat{k} \\[1.5ex]
		\vec{A} \times \vec{B} &= A_x B_x \,(0) + A_x B_y \,(+\hat{k}) + A_x B_z \,(-\hat{j}) \\[0.75ex]
		&+ A_y B_x \,(-\hat{k}) + A_y B_y \, (0)  + A_y B_z \,(+\hat{i}) \\[0.75ex]
		&+ A_z B_x \,(+\hat{j}) + A_z B_y \,(-\hat{i}) + A_z B_z \,(0) \\[1.5ex]
		\vec{A} \times \vec{B} &= A_x B_y \,\hat{k} - A_x B_z \,\hat{j} - A_y B_x \,\hat{k} + A_y B_z \,\hat{i} + A_z B_x \,\hat{j} - A_z B_y \,\hat{i} \\[1.5ex]
		\vec{A} \times \vec{B} &= \left(A_y B_z - A_z B_y \right) \,\hat{i} + \left( A_z B_x - A_x B_z \right) \,\hat{j} + \left(A_x B_y - A_y B_x  \right)\,\hat{k} \\[1.5ex]
	\end{aligned}   $$



This is how the cross product is calculated in coordinate form. However, this method is long, somewhat complicated, and certainly tedious. Is there an easier way to perform this operation? Yes—by using a determinant! Though, this trick only works in three dimensions.




### Method 1: Determinate of a $3\times 3$ Matrix

To do this, first set up a matrix where the first row consists of the unit vectors $ \hat{i} $, $ \hat{j} $, and $ \hat{k} $. The second row will be the components of whichever vector comes first in the cross product, and the third row will contain the components of the second vector. For example, for the cross product $ \vec{A} \times \vec{B} $, the matrix setup would look like this:

$$
\begin{bmatrix}
	\hat{i} & \hat{j} & \hat{k} \\
	A_x & A_y & A_z \\
	B_x & B_y & B_z 
\end{bmatrix}
$$

Now, use the cofactor expansion along the first row to compute the determinant:

$$
\begin{vmatrix}
	\hat{i} & \hat{j} & \hat{k} \\
	A_x & A_y & A_z \\
	B_x & B_y & B_z 
\end{vmatrix} = \hat{i} \begin{vmatrix} A_y & A_z \\ B_y & B_z \end{vmatrix} - \hat{j} \begin{vmatrix} A_x & A_z \\ B_x & B_z \end{vmatrix} + \hat{k} \begin{vmatrix} A_x & A_y \\ B_x & B_y \end{vmatrix}
$$

Expanding the $2\times2$ determinants gives:

$$
\begin{vmatrix}
	\hat{i} & \hat{j} & \hat{k} \\
	A_x & A_y & A_z \\
	B_x & B_y & B_z 
\end{vmatrix} = \hat{i} (A_y B_z - A_z B_y) - \hat{j} (A_x B_z - A_z B_x) + \hat{k} (A_x B_y - A_y B_x)
$$


Rearranging this result, we see that it matches the component form of the cross product we derived earlier:


$$ \begin{vmatrix}
	\hat{i} & \hat{j} & \hat{k} \\
	A_x & A_y & A_z \\
	B_x & B_y & B_z 
\end{vmatrix} = \left( A_y B_z - A_z B_y \right) \, \hat{i} + \left( A_z B_x - A_x B_z \right) \, \hat{j} +  \left( A_x B_y - A_y B_x \right) \, \hat{k} = \vec{A} \times \vec{B}  $$

This leaves us with a useful shortcut for calculating the cross product in 3 dimensions.

{% capture ex %}
In 3 dimensions, the cross product between two vectors can be calculated as follows:

$$
\vec{A} \times \vec{B} = \begin{vmatrix}
	\hat{i} & \hat{j} & \hat{k} \\
	A_x & A_y & A_z \\
	B_x & B_y & B_z 
\end{vmatrix}
$$

{\color{red} **Warning!**} This trick only works in 3 dimensions! Calculating cross products in any space other than 3 dimensions requires you to use the component representation instead.

{% endcapture %}
{% include example.html content=ex %}


{% capture ex %}
Suppose you have a charged particle being acted on by an electromagnetic force. The electric and magnetic fields are given as:

$$ \vec{E} = E \, \hat{i} \qquad \vec{B} = B\,\hat{j} $$

The electromagnetic force on the charged particle can be written using the Lorentz Force equation as:
$$ \vec{F}^{EM} = q \left( \vec{E} + \vec{v} \times \vec{B}  \right)  $$
where $\vec{v}$ is the velocity of the particle. Assuming the particle is moving in all three dimensions, we can calculate the cross product using the determinant trick:
$$ \vec{v} \times \vec{B}  = \begin{vmatrix}
	\hat{i} & \hat{j} & \hat{k} \\
	v_x & v_y & v_z \\
	0 & B & 0 
\end{vmatrix} = - B \begin{vmatrix}
\hat{i} & \hat{k} \\
v_x &  v_z
\end{vmatrix} = - B (v_z \hat{i} - v_x \hat{k} ) = - B v_z \, \hat{i} + B v_x \,\hat{k}  $$
where we have use the cofactor expansion technique along the first row.

Putting this all together gives us:
$$ \vec{F}^{EM} = q \left( E \,\hat{i}  - B v_z \, \hat{i} + B v_x \,\hat{k}  \right) \implies  \vec{F}^{EM} = (q E - q B v_z) \, \hat{i} + q B v_x \,\hat{k}  $$
{% endcapture %}
{% include example.html content=ex %}

{% capture ex %}
The angular momentum, defined for a point particle as:
$$ \vec{L} = \vec{r} \times \vec{p} $$
where $\vec{r}$ is the position of the particle relative to some origin, and $\vec{p}$ is the momentum of the particle. The angular momentum is important in almost all aspects of physics, particularly in orbital mechanics.

Consider a planet of mass $ m $ orbiting a star at a distance $ r $ with a velocity $ \vec{v} $ that is perpendicular to the position vector $ \vec{r} $ from the star to the planet. We’ll assume a circular orbit for simplicity.

The angular momentum $ \vec{L} $ of the planet is given by:
$$
\vec{L} = \vec{r} \times \vec{p}
$$
where $ \vec{p} $ is the linear momentum of the planet, defined as:
$$
\vec{p} = m \vec{v}
$$

Since $ \vec{v} $ is perpendicular to $ \vec{r} $ in a circular orbit, the magnitude of the cross product simplifies to:
$$
\lvert\vec{L}\rvert = \lvert\vec{r}\rvert \lvert\vec{p}\rvert \sin(\theta)
$$
where $ \theta = 90^\circ $, so $ \sin(\theta) = 1 $. Therefore:
$$
\lvert\vec{L}\rvert = r \, m v
$$

Using this, we can rewrite the angular momentum as:
$$
\vec{L} = r \, m v \, \hat{n}
$$
where $ \hat{n} $ is a unit vector perpendicular to the plane of $ \vec{r} $ and $ \vec{v} $ (following the right-hand rule).

For a planet of mass $ m = 5.98 \times 10^{24} \, \text{kg} $ orbiting a star at a radius $ r = 1.5 \times 10^{11} \, \text{m} $ with a velocity $ v = 3 \times 10^4 \, \text{m/s} $, the angular momentum would be:

$$
\lvert\vec{L}\rvert = (5.98 \times 10^{24} \, \text{kg}) \cdot (3 \times 10^4 \, \text{m/s}) \cdot (1.5 \times 10^{11} \, \text{m}) = 2.69 \times 10^{40} \, \text{kg} \cdot \text{m}^2/\text{s}
$$

with direction of $ \vec{L} $ being given by the right-hand rule.

{% endcapture %}
{% include example.html content=ex %}










## Mixed Products

In physics we often come upon vector operations that are a mixture of dot and cross products. Let's take a moment to consider the most common occurrences. 




### Scalar Triple Product

The **scalar triple product**, defined as:
$$
\vec{A} \cdot (\vec{B} \times \vec{C}),
$$
produces a scalar value and has a straightforward geometric interpretation.

1. **Volume of a Parallelepiped**: Geometrically, the scalar triple product represents the volume of a parallelepiped (a 3 dimensional wonky box) that has a set of edges emanating from one corner of the box defined by the vectors $\vec{A}$, $\vec{B}$, and $\vec{C}$. 

	$$
	\text{Volume} = \lvert\vec{A} \cdot (\vec{B} \times \vec{C})\rvert
	$$

	This also means that the magnitude of the cross product defined the area spanned by the wonky rectangle created by the two vectors involved in the cross product. 

2. **Cyclic Property**: The scalar triple product is invariant under a cyclic permutation:

	$$
	\vec{A} \cdot (\vec{B} \times \vec{C}) = \vec{B} \cdot (\vec{C} \times \vec{A}) = \vec{C} \cdot (\vec{A} \times \vec{B}).
	$$

3.  **Orientation and Sign**: The sign of the scalar triple product indicates whether the set of vectors $(\vec{A}, \vec{B}, \vec{C})$ forms a right-handed or left-handed system. This is helpful in understanding orientation in three-dimensional space.

In electromagnetism, the scalar triple product often appears in calculating flux through a volume or when defining handedness in coordinate systems.


{% capture ex %}
Consider three vectors: 
$$ \vec{A} = 3 \hat{i} + 2 \hat{j} - \hat{k} \qquad \vec{B} = -\hat{i} + 4 \hat{j} + 2 \hat{k}  \qquad   \vec{C} = \hat{i} - \hat{j} + 5 \hat{k} $$ 
The volume $ V $ of a parallelepiped formed by three vectors will be given by the scalar triple product:
$$
V = \lvert\vec{A} \cdot (\vec{B} \times \vec{C})\rvert
$$

To find the volume, we must first calculate the cross product $ \vec{B} \times \vec{C} $

$$
\begin{aligned}
	\vec{B} \times \vec{C} &= \begin{vmatrix}
		\hat{i} & \hat{j} & \hat{k} \\
		-1 & 4 & 2 \\
		1 & -1 & 5
	\end{vmatrix}  \\
	&= \hat{i} \begin{vmatrix} 4 & 2 \\ -1 & 5 \end{vmatrix} - \hat{j} \begin{vmatrix} -1 & 2 \\ 1 & 5 \end{vmatrix} + \hat{k} \begin{vmatrix} -1 & 4 \\ 1 & -1 \end{vmatrix}
		\\
	&= \hat{i} (4 \cdot 5 - 2 \cdot (-1)) - \hat{j} (-1 \cdot 5 - 2 \cdot 1) + \hat{k} (1 - 4)\\
	\vec{B} \times \vec{C} &= 22 \hat{i} + 7 \hat{j} - 3 \hat{k}
\end{aligned}
$$

Now, take the dot product of $ \vec{A} = 3 \hat{i} + 2 \hat{j} - \hat{k} $ with the result of $ \vec{B} \times \vec{C} = 22 \hat{i} + 7 \hat{j} - 3 \hat{k} $:

$$
\begin{aligned}
	\vec{A} \cdot (\vec{B} \times \vec{C}) &= (3 \hat{i} + 2 \hat{j} - \hat{k}) \cdot (22 \hat{i} + 7 \hat{j} - 3 \hat{k})  \\
	&= (3 \cdot 22) + (2 \cdot 7) + (-1 \cdot -3)  \\
	&= 66 + 14 + 3  \\
	\vec{A} \cdot (\vec{B} \times \vec{C}) &= 83
\end{aligned}
$$

Thus, the volume of the parallelepiped formed by the vectors $ \vec{A} $, $ \vec{B} $, and $ \vec{C} $ is the magnitude of the scalar triple product, which is  83 in whatever units are being used.

{% endcapture %}
{% include example.html content=ex %}







### Vector Triple Product

The **vector triple product**, expressed as:
$$
\vec{A} \times (\vec{B} \times \vec{C}),
$$
results in a vector and can be simplified using the following identify known as ``BAC-CAB":
$$
\vec{A} \times (\vec{B} \times \vec{C}) = \vec{B} (\vec{A} \cdot \vec{C})  - \vec{C} (\vec{A} \cdot \vec{B}) 
$$
where the name ``BAC-CAB" comes from the order of the result then the triple product is written as given above. 

The vector triple product is encountered in various areas of physics, particularly in electromagnetism and rigid body dynamics. In electromagnetism, it plays a role in the derivation of the Biot-Savart Law and is useful for calculating magnetic forces, where multiple cross products appear in the expressions for magnetic fields and forces on moving charges. Additionally, in rigid body dynamics, the vector triple product identity simplifies expressions involving torque and angular momentum, especially when analyzing motion in non-inertial or rotating reference frames.


{% capture ex %}
Consider a **current loop** placed in a magnetic field. The force on a segment of current $ I \, \mathrm{d}\vec{\ell} $ in a magnetic field $ \vec{B} $ is given by:
$$
\mathrm{d}\vec{F} = I \, \mathrm{d}\vec{\ell} \times \vec{B}
$$

If we consider the torque $ \vec{\tau} $ exerted by this force on a loop with position vector $ \vec{r} $, we have:
$$
\vec{\tau} = \vec{r} \times (I \, \mathrm{d}\vec{\ell} \times \vec{B})
$$

Using the vector triple product identity, this expression simplifies to:
$$
\vec{\tau} = (I \, \mathrm{d}\vec{\ell}) \,  (\vec{r} \cdot \vec{B})  - \vec{B} \, (\vec{r} \cdot I \, \mathrm{d}\vec{\ell}) 
$$

This form is particularly useful in analyzing the torque on magnetic dipoles or current-carrying coils in non-uniform magnetic fields.

{% endcapture %}
{% include example.html content=ex %}





## Application:

Consider a magnetic dipole moment $ \vec{\mu} $ (this could be a particle with a non-zero spin or an atom with a magnetic momentum) placed in a uniform magnetic field $ \vec{B} $. The torque $ \vec{\tau} $ exerted on the dipole is given by the cross product of the magnetic dipole moment and the magnetic field:
$$
\vec{\tau} = \vec{\mu} \times \vec{B}
$$
Additionally, the potential energy $ U $ of the magnetic dipole in a magnetic field is given by:
$$
U = - \vec{\mu} \cdot \vec{B}
$$
Suppose the angle vector $ \vec{\theta} $ represents the angle the magnetic moment is forces to rotate through via some external input.

Given the following vectors:

- $ \vec{\mu} = 2 \hat{i} + 3 \hat{j} + \hat{k} \, \text{Am}^2 $ (magnetic dipole moment),
- $ \vec{B} = \hat{i} - 2 \hat{j} + 4 \hat{k} \, \text{T} $ (magnetic field),
- $ \vec{\theta} = \hat{i} + 3 \hat{j} - 2 \hat{k} $ (angle rotation vector in radians)


Let's find the following: 


1. The torque $ \vec{\tau} $ exerted on the dipole.
	
	The torque is given by the cross product:

	$$
	\vec{\tau} = \vec{\mu} \times \vec{B}
	$$

	Since this is 3 dimensions, we can take this cross product via the determinant method. Substituting the given values of $ \vec{\mu} $ and $ \vec{B} $ into the matrix and taking the determinant gives:

	$$
	\vec{\tau} = \begin{vmatrix}
		\hat{i} & \hat{j} & \hat{k} \\
		2 & 3 & 1 \\
		1 & -2 & 4
	\end{vmatrix} = \hat{i} (12 + 2) - \hat{j} (8 - 1) + \hat{k} (-4 - 3) = (14 \hat{i} - 7 \hat{j} - 7 \hat{k}) \, \text{N} \cdot \text{m}
	$$
	
	
2. The potential energy of the magnetic dipole in the magnetic field.
	Substituting the given vectors into the definition of the potential energy of the magnetic dipole in an external magnetic field gives:
	
	$$
	U = - \left( 2 \cdot 1 + 3 \cdot (-2) + 1 \cdot 4 \right) = 0 \, \text{J}
	$$
	
3. The work done by the magnetic field on the dipole as it is rotate according to $\vec{\theta}$.
	The work can be found by tkaing the dot product between $\vec{\theta}$ and $\vec{\tau}$:
	
	$$
	W = \vec{\theta} \cdot \vec{\tau} =  \vec{\theta} \cdot \left(\vec{\mu} \times \vec{B}\right)
	$$
	
	Substituting the cross product $ \vec{\mu} \times \vec{B} = 14 \hat{i} - 7 \hat{j} - 7 \hat{k} $:
	
	$$
	W = \left( \hat{i} + 3 \hat{j} - 2 \hat{k}\right) \cdot \left(14 \hat{i} - 7 \hat{j} - 7 \hat{k}\right) = 14 - 21 + 14 = 7\,\text{J}
	$$








## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


Consider an electric dipole with dipole moment $ \vec{p} $ (this could be a molecule like H$_2$O or something that inherently has an electric dipole) placed in a uniform electric field $ \vec{E} $. The torque $ \vec{\tau} $ exerted on the dipole is given by the cross product of the electric dipole moment and the electric field:

$$
\vec{\tau} = \vec{p} \times \vec{E}
$$

Additionally, the potential energy $ U $ of the electric dipole in an electric field is given by:

$$
U = - \vec{p} \cdot \vec{E}
$$

Suppose an external force applies an angular displacement vector $ \vec{\alpha} $, which represents the angle (in radians) through which the dipole rotates in the plane of the electric field.

Given the following vectors:

- $ \vec{p} = 4 \hat{i} - 2 \hat{j} + 3 \hat{k} \, \text{Cm} $ (electric dipole moment),
- $ \vec{E} = 2 \hat{i} + \hat{j} - 3 \hat{k} \, \text{N/C} $ (electric field),
- $ \vec{\alpha} = -\hat{i} + 2 \hat{j} + \hat{k} $ (rotation vector in radians)


Find the following: 

 a) The torque $ \vec{\tau} $ exerted on the electric dipole.  

b) The potential energy $ U $ of the electric dipole in the electric field.  

c)  The work $ W $ done by the electric field on the dipole as it rotates through the angle $ \vec{\alpha} $.
