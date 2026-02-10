---
layout: default
title: Mathematical Methods - Short Homework 10
course_home: /courses/math-methods/
nav_section: homework
hw_type: short
nav_order: 13
---

## PHYS 3345 - Short Homework 10

---

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.

---


In quantum mechanics, the Hamiltonian operator $ \hat{H} $ determines the total energy of a system. Suppose we are working with a two-level quantum system, such as a spin-$\frac{1}{2}$ particle in an external magnetic field. The Hamiltonian is represented by the following $ 2 \times 2 $ matrix:

$$
\mathbf{H} = \begin{bmatrix}
	\Delta & \gamma \\
	\gamma & -\Delta
\end{bmatrix}
$$

where:


- $ \Delta $ is a real number representing the energy splitting of the two levels in the absence of interactions.
- $ \gamma $ is a real coupling constant describing the interaction between the levels.
- Row 1, Column 1 corresponds to state 1 of the quantum system with an energy splitting given by $\Delta$. 
- Row 2, Column 2 corresponds to state 2 of the quantum system with an energy splitting given by $-\Delta$. 
- Row 1, Column 2 represents the shift in energy of state 1 due to possible interactions coming from state 2.
- Row 2, Column 1 represents the shift in energy of state 2 due to possible interactions coming from state 1.


In the case $\gamma \ne 0$ we would say states 1 and 2 are coupled together. In this problem we will see if we can find a representation where our new working states are decoupled. 

For this problem, you can take $\Delta = 3$ and $\gamma = 4$. 


a) Find the eigenvalues of the Hamiltonian matrix $ \mathbf{H} $. What do you think these eigenvalues represent in the context of the quantum system?  
b) Determine the eigenvectors of $ \mathbf{H} $.   
c) Construct the matrix $ \mathbf{P} $, whose columns are the eigenvectors of $ \mathbf{H} $, and verify that $ \mathbf{P}^{-1} \mathbf{H} \mathbf{P} $ is diagonal. What are the diagonal entries of this matrix, and what do you think they represent?


