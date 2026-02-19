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


### 1) Some Quick Definitions and Concepts

a) Define a complex number in rectangular form and express it in polar form. Make up some numbers, and plot the vector representation of the complex number you created.   
	
b) Define what an eigenvalue and an eigenvector are for a given square matrix. Discuss briefly why these concepts could be important in modeling physical systems.  
	
c) Define what is meant by basis vectors in a vector space. Additionally, explain the concept of linear independence and how the determinant of a matrix can be used to test for it.  






### 2) Complex Numbers and Matrices

a) Using what we found in lecture, show that a complex number $ z = a + bi $ can be represented as a $2 \times 2$ matrix:  

$$
\mathbf{Z} = \begin{bmatrix} a & -b \\ b & a \end{bmatrix}
$$

b) Verify that multiplying two complex numbers corresponds to multiplying their matrix representations. That is, if   

$$
z_1 = a + bi \quad \text{and} \quad z_2 = c + di
$$

show that:

$$
z_1 z_2 = \text{matrix product of } \begin{bmatrix} a & -b \\ b & a \end{bmatrix} \text{ and } \begin{bmatrix} c & -d \\ d & c \end{bmatrix}
$$



### 3) Matrix Categorization

#### Matrix $ \mathbf{A} $ (Real):

$$
\mathbf{A} = \begin{bmatrix}
   2 & -1 \\
   -1 & 3
\end{bmatrix}
$$

Is this matrix...

a) ...  symmetric or antisymmetric?
b) ... orthogonal (i.e., $\mathbf{A}^T\mathbf{A} = \mathbf{I}$)?

#### Matrix $ \mathbf{B} $ (Complex):

$$
\mathbf{B} = \begin{bmatrix}
   0 & 1+i \\
   1-i & 0
\end{bmatrix}
$$

Is this matrix...

c) ...  Hermitian (i.e., $\mathbf{B} = \mathbf{B}^\dagger$) or Anti-Hermitian (i.e., $\mathbf{B} = -\mathbf{B}^\dagger$).  
d) ... unitary (i.e., $\mathbf{B}^\dagger\mathbf{B} = \mathbf{I}$).  


### 4) Eigenvalue and Eigenvector Analysis for a 2x2 Matrix

a) Consider the matrix:

$$
\mathbf{A} = \begin{bmatrix} 3 & 2 \\ 2 & 3 \end{bmatrix}
$$

Calculate the eigenvalues of $ \mathbf{A} $ by finding the roots of the characteristic equation.

b) For each eigenvalue, determine the corresponding eigenvector.

c) Are these eigenvectors orthogonal?






### 5) Application in Quantum Mechanics

a) Consider a Hamiltonian operator for a two-level quantum system represented by the matrix:

$$
\mathbf{H} = \begin{bmatrix} 2 & 1 - i \\ 1 + i & 3 \end{bmatrix}
$$

Calculate the eigenvalues and eigenvectors of $ \mathbf{H} $.

b) Discuss the physical significance of the eigenvalues in the context of energy levels in quantum mechanics. (This is graded on attempt, so do not panic if you have not had Quantum Mechanics yet!)





### 6) Eigenvalue and Eigenvector Practice

Consider the matrix

$$
\mathbf{E} = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}
$$

a) Derive the characteristic equation for $\mathbf{E}$ and compute its eigenvalues.  
b) For each eigenvalue, find a corresponding eigenvector.  
c) Verify that your eigenvectors are linearly independent.  


<div class="page-break"></div>


### 7) Eigenvalue and Eigenvector Practice... again!

Consider the matrix

$$
\mathbf{F} = \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix}
$$

a) Calculate the eigenvalues of $\mathbf{F}$.  
b) Determine the eigenvectors corresponding to each eigenvalue.  






