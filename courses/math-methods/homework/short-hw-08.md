---
layout: default
title: Mathematical Methods - Short Homework 08
course_home: /courses/math-methods/
nav_section: homework
hw_type: short
nav_order: 11
---

## PHYS 3345 - Short Homework 08

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


Consider a symmetric rigid body rotating freely about its center of mass. Its rotational dynamics are governed by the moment of inertia tensor, $\mathbf{I}$, and the angular velocity vector, $\vec{\omega}$. Assume the body has principal moments of inertia $I_1$, $I_2$, and $I_3$, such that the matrix representation of $\mathbf{I}$ in the principal axis system is diagonal:

$$
\mathbf{I} =
\begin{bmatrix}
	I_1 & 0 & 0 \\
	0 & I_2 & 0 \\
	0 & 0 & I_3
\end{bmatrix}.
$$

The stability of rotation about a principal axis is determined by the eigenvalues of the following matrix:

$$
\mathbf{A} =
\begin{bmatrix}
	0 & \frac{I_3 - I_1}{I_1} \omega_3 & \frac{I_2 - I_1}{I_1} \omega_2 \\
	\frac{I_3 - I_2}{I_2} \omega_3 & 0 & \frac{I_1 - I_2}{I_2} \omega_1 \\
	\frac{I_2 - I_3}{I_3} \omega_2 & \frac{I_1 - I_3}{I_3} \omega_1 & 0
\end{bmatrix}
$$

a)   For $I_1 = 2$, $I_2 = 3$, $I_3 = 5$, and $\omega_1 = 1$, $\omega_2 = 2$, $\omega_3 = 3$, find $\mathbf{A}$ for these numbers.  
b) Set up the characteristic equation for matrix $\mathbf{A}$.   
c) Find the eigenvalues for matrix $\mathbf{A}$.   















