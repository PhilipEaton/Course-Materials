---
layout: default
title: Mathematical Methods - Long Homework 06
course_home: /courses/math-methods/
nav_section: homework
hw_type: long
nav_order: 18
---

## PHYS 3345 - Long Homework 06

---

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.

---

### 1) Solving with the Method of Undetermined Coefficients

Solve the following second-order linear inhomogeneous ODEs using the method of undetermined coefficients. Be sure to first find the homogeneous solution, so you can make a good guess when applying the method of undetermined coefficients to find the particular solution. Lastly, write the general solution.

a) $ y'' + 3y' + 2y = 5e^x $  
b) $ y'' - 4y = \sin(2x) $  








### 2) Variation of Parameters Practice

Use the method of variation of parameters to find the general solution to the differential equation:

$$
y'' + y = \tan x
$$

a) Solve for the homogeneous solution.  
b) Use variation of parameters to find a particular solution.  
c) Write the general solution.  







### 3) Laplace Transforms and Solving ODEs

Use Laplace transforms to solve the following initial value problems. Show each step: the transform, algebraic manipulation, and the inverse transform.

a) $ y'' + 4y = 0, \quad y(0) = 0, \quad y'(0) = 1 $  
b) $ y'' + 2y' + y = e^{-t}, \quad y(0) = 0, \quad y'(0) = 0 $  
c) $ y'' + y = \delta(t - \pi), \quad y(0) = 0, \quad y'(0) = 0 $  







### 4) Reduction of Order

Given that $ y_1(x) = x $ is a solution to the homogeneous equation

$$
x^2 y'' - 3x y' + 4y = 0
$$

use reduction of order to find a second linearly independent solution.







### 5) Mixed Concept Problem

Solve the following initial value problem using Laplace transforms. Then verify your solution matches the long-time behavior predicted by steady-state analysis.

$$
y'' + 2y' + y = U(t - 1), \quad y(0) = 0, \quad y'(0) = 0
$$







### 6) Solving a System of ODEs with Eigenvalues

Solve the following system of ODEs by finding the eigenvalues and eigenvectors of the coefficient matrix:

$$ \frac{d}{dt} \begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix} \begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

Assume $ \vec{x}(0) = \begin{bmatrix} 1 \\ 0 \end{bmatrix} $. Find the general solution and apply the initial condition.





### 7) Modeling a Coupled Oscillator System

A system of two masses connected by springs is governed by:

$$
\begin{aligned}
m_1 \frac{d^2 x_1}{dt^2} &= -k(x_1 - x_2) \\
m_2 \frac{d^2 x_2}{dt^2} &= -k(x_2 - x_1)
\end{aligned}
$$

a) Rewrite this as a second-order system using matrix notation.  
b) Let $ m_1 = m_2 = 1 $, $ k = 2 $. Find the general solution for this model.  





