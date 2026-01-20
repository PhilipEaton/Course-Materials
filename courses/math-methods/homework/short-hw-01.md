---
layout: default
title: Mathematical Methods - Short Homework 01
course_home: /courses/math-methods/
nav_section: homework
nav_order: 1
---


## PHYS 3345 - Short Homework 01

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.

### Problem 1:

Consider the following matrices:

$$
\mathbf{A} = \begin{bmatrix} 2 & 3 \\ 1 & 4 \\ -1 & 2 \end{bmatrix} \quad \mathbf{B} = \begin{bmatrix} 5 & -2 \\ 3 & 6 \end{bmatrix}
$$

a) Multiply the matrices \\(\mathbf{A}\\) and \\(\mathbf{B}\\) to find the resulting matrix \\(\mathbf{C} = \mathbf{A} \mathbf{B}\\). If this operation is not allowed, explain why.  

b) If the matrix multiplication is permitted, determine the size of the resulting matrix \\(\mathbf{C}\\). Does this result agree with the rules established earlier?  

c) Multiply the matrices \\(\mathbf{B}\\) and \\(\mathbf{A}\\) to find the resulting matrix \\(\mathbf{D} = \mathbf{B} \mathbf{A}\\). If this operation is not allowed, explain why.  

d) If the matrix multiplication is permitted, determine the size of the resulting matrix \\(\mathbf{D}\\). Does this result agree with the rules established earlier?  


### Problem 2:

Consider the following circuit with three loops and three resistors. The circuit contains two voltage sources, \\( V_1 = 10 \, \text{V} \\) and \\( V_2 = 5 \, \text{V} \\), and three resistors with values \\( R_1 = 2 \, \Omega \\), \\( R_2 = 3 \, \Omega \\), and \\( R_3 = 4 \, \Omega \\).

Using Kirchhoff's Voltage Law, we obtain the following system of equations for the currents \\( I_1 \\), \\( I_2 \\), and \\( I_3 \\) flowing through each loop:

$$
\begin{aligned}
	2 I_1 + 3 I_2 &= 10, \\
	-2 I_1 + 4 I_3 &= 5, \\
	3 I_2 - 4 I_3 &= -5.
\end{aligned}
$$

a) Write this system of linear equations in matrix form, \\(\mathbf{A} \vec{I} = \vec{V}\\), where \\(\mathbf{A}\\) is the matrix of coefficients, \\(\vec{I}\\) is the vector of unknown currents, and \\(\vec{V}\\) is the vector of voltage values.  

b) Write out the augmented matrix for this system of linear equations. (Remember to include 0's if a variable is not present in an equation!)	  

c) Solve for the currents \\( I_1 \\), \\( I_2 \\), and \\( I_3 \\) using the Gaussian elimination process described in the Application section.  

d) Interpret your solution: what do the values of \\( I_1 \\), \\( I_2 \\), and \\( I_3 \\) indicate about the direction and magnitude of currents in each loop?  