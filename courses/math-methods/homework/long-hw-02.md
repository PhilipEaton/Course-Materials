---
layout: default
title: Mathematical Methods - Long Homework 02
course_home: /courses/math-methods/
nav_section: homework
hw_type: long
nav_order: 8
---

## PHYS 3345 - Long Homework 02

---

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.

---


### 1) Some Quick Definitions and Concepts

a) *Symmetric and Antisymmetric Matrices:*
   - Define a symmetric matrix and provide an example.
   - Define an antisymmetric (aka skew-symmetric) matrix and provide an example.

b) *Orthogonal Matrices:*  
   - Define an orthogonal matrix.
   - List its defining properties.

c) *Determinants and Linear Independence:*
   - Explain how the determinant of a matrix formed by a set of vectors can be used to determine if those vectors are linearly independent.
   - Provide a brief example to illustrate this idea. (Hint, make up two 2D vectors that are dependent and show you test agrees they are indeed dependent.)






### 2) Matrix Inversion
Given the matrix

$$
\mathbf{A} = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}
$$

a) Compute the inverse $ \mathbf{A}^{-1} $ using either row reduction methods or the formula given in the notes for a $2 \times 2$ matrix.  
b) Verify your result by calculating $ \mathbf{A} \mathbf{A}^{-1} $ and $ \mathbf{A}^{-1} \mathbf{A} $. What should you get from there operations?



<div class="page-break"></div>

### 3) Symmetric-Antisymmetric Decomposition 
Decompose the matrix

$$
\mathbf{B} = \begin{bmatrix} 3 & 4 \\ 4 & 1 \end{bmatrix}
$$

into its symmetric and antisymmetric components.




### 4) Testing Linear Independence 
Consider the vectors:

$$
\vec{v}_1 = \begin{bmatrix} 2 \\ 0 \\ 0 \end{bmatrix} \quad\quad
\vec{v}_2 = \begin{bmatrix} 0 \\ 3 \\ 0 \end{bmatrix} \quad\quad
\vec{v}_3 = \begin{bmatrix} 0 \\ 0 \\ 4 \end{bmatrix}
$$

a) Form the matrix with these vectors as columns and compute its determinant and use the result to discuss the linear independence of these vectors.  
b) Now, assume a constraint forces the system to move only in the $yz$-plane. Update the vectors accordingly, and determine the rank of the new matrix. (Hint: If the system is constrained to move only in the $yz$-plane, then can it move in the $x$-direction?)



### 5) Verifying Orthogonality
Given the matrix

$$
\mathbf{R} = \begin{bmatrix} \frac{2^{1/2}}{2} & -\frac{2^{1/2}}{2} \\ \frac{2^{1/2}}{2} & \frac{2^{1/2}}{2} \end{bmatrix}
$$

a) Verify that $ \mathbf{R} $ is orthogonal by showing that $ \mathbf{R}^\text{T} \mathbf{R} = \mathbf{I} $.  
b) Compute the determinant of $ \mathbf{R} $ and explain its significance.  







### 6) Combining Multiple Transformations
An object in the $xy$-plane is first rotated by $45^\circ$ using the rotation matrix $\mathbf{R}(45^\circ)$ **and then** reflected about the $y$-axis using $\mathbf{R}_y$. Both matrices can be given as:

$$
\mathbf{R}(45^\circ) = \begin{bmatrix} \cos(45^\circ) & -\sin(45^\circ) \\ \sin(45^\circ) & \cos(45^\circ) \end{bmatrix} \qquad \qquad \mathbf{R}_y = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}
$$


a) Compute the combined transformation matrix. (Hint: Make sure the order is correct!)  
b) Apply the combined transformation to the vector 

$$ \vec{v} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} $$
 
and confirm the transformation was completed correctly.  

c) Reverse the order of the transformations, create combined transformation matrix, and apply it to vector 

$$ \vec{v} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} $$ 

Is this the same or different as what you got in part (b). Does that make sense? 




### 7) Basis and Rank
A set of data points is represented by the vectors:

$$
\vec{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \quad\quad
\vec{v}_2 = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} \quad\quad
\vec{v}_3 = \begin{bmatrix} 7 \\ 8 \\ 9 \end{bmatrix}
$$


a) Form the matrix with these vectors as columns and determine the rank of the matrix.  
b) Explain what the rank indicates about the dimensionality of the space spanned by the data.  















