---
layout: default
title: Mathematical Methods - Long Homework 03
course_home: /courses/math-methods/
nav_section: homework
hw_type: long
nav_order: 12
---

## PHYS 3345 - Long Homework 03

---

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.

---


### 1) Row Reduction Makes Me Cry
Consider the system of equations:  

$$
\begin{aligned}
	2x + 3y - z &= 5 \\
	x - 4y + 2z &= -3 \\
	-3x + 2y + 4z &= 7
\end{aligned}
$$

a) Write this system in matrix form $ \mathbf{A} \vec{x} = \vec{b} $, where $ \mathbf{A} $ is the coefficient matrix, $ \vec{x} $ is the variable vector, and $ \vec{b} $ is the result vector.  

b) Solve for $x$, $y$, and $z$, by setting up the augmented matrix and using row reduction to find the solution.  

c) Find the inverse of the coefficient matrix (you can do this on the computer if you want) and use that to find $x$, $y$, and $z$.

<br>



### 2) Determinants and Their Meaning

Given the matrix 

$$ 
\mathbf{A} = \begin{bmatrix} 2 & 1 & 3 \\ -1 & 4 & 2 \\ 0 & -2 & 5 \end{bmatrix} 
$$

a) Calculate $ \det(\mathbf{A}) $.  

b) Interpret what the determinant tells you about the matrix $ \mathbf{A} $ in terms of:
   - Whether $ \mathbf{A} $ is has an inverse or not.
   - Whether $ \mathbf{A} $ changes the volume of a unit cube. If so, by what factor?




### 3) Transpose and Trace
Consider the matrix 

$$ 
\mathbf{B} = \begin{bmatrix} 4 & 7 \\ -2 & 3 \end{bmatrix} 
$$

Compute $ \mathbf{B}^\text{T} $ and tind the trace of $ \mathbf{B} $.

<div class="page-break"></div>


### 4) Coordinate Transformations
a)}] A particle is traveling with a velocity of 

$$
\vec{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix} 
$$ 

Somehow, via magnets or magic, it undergoes a rotation of $ 90^\circ $ counterclockwise about the origin. Write the rotation matrix and compute the new components for the velocity. Call the resulting velocity $ \vec{v}\,' $.  

b) Reflect $ \vec{v}\,' $ off the line $ y = x $, and provide the resulting coordinates.  

c) Discuss whether the transformations preserve the magnitude of the vector. Does that make sense considering the determinants of each of the transformation matrices?




### 5) Dot Product and Work
A constant force $ \vec{F} = 3\hat{i} - 4\hat{j} \, \text{N} $ is applied to an object that moves with displacement $ \Delta \vec{x} = 5\hat{i} + 2\hat{j} \, \text{m} $.

a) Compute the work done by the force using the dot product.  
b) If the displacement were not the one given but was actually perpendicular to the force, how this would affect the resulting value for the work done?







### 6) Cross Product
Two vectors are given as $ \vec{A} = 2\hat{i} + 3\hat{j} + \hat{k} $ and $ \vec{B} = \hat{i} - 4\hat{j} + 2\hat{k} $.

a) Compute $ \vec{A} \times \vec{B} $.  
b) Verify that $ \vec{A} \cdot (\vec{A} \times \vec{B}) = 0 $.  
c) Show that the cross product $ \vec{A} \times \vec{B} $ can be expressed as $ \mathbf{A}_\times \vec{B} $, where:

$$
\mathbf{A}_\times = \begin{bmatrix}
   0 & -A_z & A_y \\
   A_z & 0 & -A_x \\
   -A_y & A_x & 0
\end{bmatrix}
$$

using the two given vectors.





### 7) Pure Scaling Transformation
A matrix $ \mathbf{S} = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} $ represents a scaling transformation.  

a) Apply $ \mathbf{S} $ to the vector $ \vec{u} = \begin{bmatrix} 4 \\ 5 \end{bmatrix} $, and find the resulting vector.  
b) Discuss how the scaling affects the area of a unit square in the plane. (Hint: The determinant may be helpful here.)















