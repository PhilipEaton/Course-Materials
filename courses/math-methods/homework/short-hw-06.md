---
layout: default
title: Mathematical Methods - Short Homework 06
course_home: /courses/math-methods/
nav_section: homework
nav_order: 7
---

## PHYS 3345 - Short Homework 06

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







