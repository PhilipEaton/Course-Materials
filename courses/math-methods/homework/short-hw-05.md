---
layout: default
title: Mathematical Methods - Short Homework 05
course_home: /courses/math-methods/
nav_section: homework
nav_order: 6
---

## PHYS 3345 - Short Homework 05

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.



Consider a 3-dimensional rigid body with a moment of inertia tensor (a matrix) $ \mathbf{I} $, which is used to describe the relationship between the angular momentum $ \vec{L} $ and the angular velocity $ \vec{\omega} $ through the equation:

$$
\vec{L} = \mathbf{I} \vec{\omega}
$$

The following matrix representations are provided:


- The **identity matrix** $ \mathbf{I}_3 $, which represents the isotropic moment of inertia:

$$
\mathbf{I}_3 = \begin{bmatrix}
   1 & 0 & 0 \\
   0 & 1 & 0 \\
   0 & 0 & 1
\end{bmatrix}
$$
	
- A **diagonal matrix** $ \mathbf{I}_\text{diag} $, representing a symmetric rigid body:

$$
\mathbf{I}_\text{diag} = \begin{bmatrix}
   3 & 0 & 0 \\
   0 & 2 & 0 \\
   0 & 0 & 1
\end{bmatrix} \, \mathrm{kg \cdot m^2}
$$

- A **symmetric matrix** $ \mathbf{I}_\text{sym} $, which represents a rigid body where the moments of inertia along different axes are not independent:

$$
\mathbf{I}_\text{sym} = \begin{bmatrix}
   4 & 1 & 0 \\
   1 & 3 & 0 \\
   0 & 0 & 2
\end{bmatrix} \, \mathrm{kg \cdot m^2}
$$

- An **antisymmetric matrix** $ \mathbf{A} $, which represents the cross-product operation for a vector $ \vec{\omega} = \begin{bmatrix} \omega_x & \omega_y & \omega_z \end{bmatrix}^\text{T} $:

$$
\boldsymbol{\omega}_{\times}  = \begin{bmatrix}
   0 & -\omega_z & \omega_y \\
   \omega_z & 0 & -\omega_x \\
   -\omega_y & \omega_x & 0
\end{bmatrix}
$$

- An **orthogonal matrix** $ \mathbf{R} $, representing a rotation about the $ z $-axis by an angle $ \theta = \frac{\pi}{4} $:

$$
\mathbf{R}_z(\tfrac{\pi}{4}) = \begin{bmatrix}
   \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}} & 0 \\
   \tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} & 0 \\
   0 & 0 & 1
\end{bmatrix}
$$



a) Verify that the angular momentum $ \vec{L} $ is the same (ignoring the units) as the angular velocity vector when the moment of inertia is represented by the identity matrix $ \mathbf{I}_3 $, and 

$$ \vec{\omega} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} $$

b) Compute $ \vec{L} $ when the moment of inertia is $ \mathbf{I}_\text{diag} $ and the angular velocity is the same as the one given in part (a).

c) Show that $ \mathbf{I}_\text{sym} $ is symmetric by explicitly verifying $ \mathbf{I}_\text{sym} = \mathbf{I}_\text{sym}^\text{T} $. Then compute $ \vec{L} $ using the same $\vec{\omega}$ as part (a).

d) Write the cross product matrix for $\vec{\omega}$ to calculate torque vector $ \vec{\tau} = \vec{\omega} \times  \vec{L} $, where

$$ \vec{L} = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix} \qquad \qquad \vec{\omega} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} $$

e) Rotate the angular momentum vector 

$$ \vec{L} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$

using the orthogonal matrix $\mathbf{R}_z(\tfrac{\pi}{4})  $ given in the problem statement. Also, verify that $ \mathbf{R}_z(\tfrac{\pi}{4}) ^\text{T} \mathbf{R}_z(\tfrac{\pi}{4})  = \mathbf{I}_3 $, confirming the orthogonality of the rotation matrix.
