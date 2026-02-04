---
layout: default
title: Mathematical Methods - Lecture 08
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 8
---

# Lecture 08 – The Eigenvalue Problem and Eigenvalues



## Eigenvalue Problem

In this lecture, we explore one of the most important problems in linear algebra: the **eigenvalue problem**. The equation 

$$ \mathbf{A} \vec{v} = \lambda \vec{v} $$

captures a fundamental idea: a matrix $\mathbf{A}$ acting on a vector $\vec{v}$ scales it by a factor of $\lambda$ *without changing its direction*. But why should we, as physicists, care? The answer lies in the fact that eigenvalues and eigenvectors appear in some of the most important equations and physical systems, from quantum mechanics to classical mechanics, and even in stability analysis of dynamic systems.









### Mathematical Framework

Let's begin by understanding the mathematical framework of eigenvalues and eigenvectors. From there we will move on to their physical interpretations since those are generally unique depending on what calculations are being done.



#### Definition of Eigenvalues and Eigenvectors

Given a square ($n \times n$) matrix $\mathbf{A}$, a nonzero vector $\vec{v}$ is called an **eigenvector** if it satisfies the equation:  

$$ \mathbf{A} \vec{v} = \lambda \vec{v} $$  

where $\lambda$ is a scalar known as the **eigenvalue** associated with $\vec{v}$. In other words, every eigenvalue has a corresponding eigenvector. 

Typically, if $\mathbf{A}$ is an $n \times n$ matrix, you will obtain $n$ unique eigenvalues. However, this is not always the case. There are instances where eigenvalues are repeated, and for each repeated eigenvalue, there will be a corresponding set of unique (linearly independent) eigenvectors. For example, if an eigenvalue appears three times, it will have three distinct eigenvectors associated with it.

An important observation for the eigenvalue problem is that the axis described by the eigenvector $\vec{v}$ is not rotated as a result of the matrix $\mathbf{A}$ acting on it. Assuming the eigenvalue $\lambda$ is nonzero, it represents a scaling along that axis without any rotation. Specifically:  

- If $\lambda > 0$, the vector $\vec{v}$ is stretched (or compressed) along its axis. For example, see the red vector below, where the solid blue vector is the original vector before being acted on by $\mathbf{A}$.  
	
- If $\lambda < 0$, the vector $\vec{v}$ is both scaled and reflected, reversing its direction.   For example, see the dashed purple vector below, where the blue vector is the original vector before being acted on by $\mathbf{A}$.  


<img
  src="{{ '/courses/math-methods/images/lec08/Scaling1.png' | relative_url }}"
  alt="The image shows an x–y coordinate plane with the horizontal axis labeled x and the vertical axis labeled y. A blue arrow representing an original vector extends from the origin into the first quadrant. A red arrow, labeled as scaled, points in the same direction as the blue vector but is longer, showing the effect of multiplying the vector by a positive number. Red text near this arrow explains that applying a matrix to the vector with a positive scaling value makes the vector longer in the same direction. A dashed red arrow extends from the origin into the third quadrant, pointing in the opposite direction from the original vector. This arrow is labeled reflected and scaled, with red text explaining that scaling by a negative number reverses the direction and changes the length."
  style="display:block; margin:1.5rem auto; max-width:400px; width:30%;">





If the eigenvalue is zero, then the transformation maps the eigenvector $\vec{v}$ to the zero vector:  
$$ \mathbf{A} \vec{v} = \lambda \vec{v} = 0 \cdot \vec{v} = \vec{0}. $$  
In this case, $\vec{v}$ lies in the **null space** (also called the **kernel** of the eigenvalue problem) of $\mathbf{A}$. This means that $\mathbf{A}$ collapses any vector in this direction to the origin, effectively annihilating it under the transformation. Eigenvalues of zero are often result when $\mathbf{A}$ is a with singular matrices, which do not have an inverse.

Understanding whether an eigenvalue is positive, negative, or zero provides important geometric insight into the transformation $\mathbf{A}$. In physics, for example, these properties can describe stretching, compression, reflection, or even the elimination of possible states/modes in systems like quantum mechanics, vibrations, or stability analysis.











#### Finding Eigenvalues

To determine the eigenvalues $\lambda$ of a square matrix $\mathbf{A}$, we begin by rewriting the eigenvalue equation:  

$$ \mathbf{A} \vec{v} = \lambda \vec{v} $$  

in the following manner:  

$$ (\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \vec{0} $$  

where $\mathbf{I}$ is the identity matrix of the same dimension as $\mathbf{A}$, and $\vec{v}$ is the eigenvector associated with the eigenvalue $\lambda$.

**We are looking for a nontrivial solution -- i.e., $\vec{v** \neq \vec{0}$.} Let's, for a moment, assume the matrix $(\mathbf{A} - \lambda \mathbf{I})$ has an inverse. If this is the case, then we can write the following:

$$
\begin{aligned}
	(\mathbf{A} - \lambda \mathbf{I}) \vec{v} &= \vec{0} \\[0.75ex]
	(\mathbf{A} - \lambda \mathbf{I})^{-1}(\mathbf{A} - \lambda \mathbf{I}) \vec{v} &= (\mathbf{A} - \lambda \mathbf{I})^{-1} \, \vec{0}  \\[0.75ex]
	\mathbf{I} \vec{v} &=  \vec{0}  \\[0.75ex]
	\vec{v} &=  \vec{0} 
\end{aligned}
$$

But, this is a contradiction to the assumption that we are looking for a nontrivial solution -- i.e., $\vec{v} \neq \vec{0}$. This means the matrix $(\mathbf{A} - \lambda \mathbf{I})$ does not have an inverse -- which means the matrix $(\mathbf{A} - \lambda \mathbf{I})$ is a **singular matrix**. A singular matrix is defined as one that does not have an inverse, which occurs precisely when its determinant is zero. So, enforcing the condition that this matrix has a determinant of zero gives us a way to find the eigenvalues.


{% capture ex %}

To find the non-trivial solution ($\vec{v} \neq \vec{0}$) to the eigenvalue problem, 

$$ \mathbf{A} \vec{v} = \lambda \vec{v} \implies (\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \vec{0} $$  

we require:  

$$ \det(\mathbf{A} - \lambda \mathbf{I}) = 0 $$  

This is known as the **characteristic equation**, and solving it yields the eigenvalues $\lambda$. Each solution to this equation will be an eigenvalue, $\lambda$, for the given eigenvalue problem.

{% endcapture %}
{% include result.html content=ex %}
 

The characteristic equation is typically a polynomial of degree $n$ for an $n \times n$ matrix $\mathbf{A}$, meaning that up to $n$ unique eigenvalues (real or complex) may be found, depending on the specific properties of $\mathbf{A}$. These eigenvalues can be distinct or repeated, with multiplicities (repeated eigenvalues) determined by the polynomial’s roots.

{% capture ex %}

Consider the matrix 

$$
\mathbf{A} = \begin{bmatrix}
	2 & 1 \\
	1 & 3
\end{bmatrix}
$$

We want to find the eigenvalues $\lambda$ such that the equation 

$$
\mathbf{A} \vec{v} = \lambda \vec{v}
$$

holds true. First, let's set up the problem so that we can get the characteristic equation:

$$
(\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \mathbf{0}
$$

where $\mathbf{I}$ is the $2 \times 2$ identity matrix. This becomes

$$
\mathbf{A} - \lambda \mathbf{I} = \begin{bmatrix}
	2 & 1 \\
	1 & 3
\end{bmatrix} - \lambda \begin{bmatrix}
	1 & 0 \\
	0 & 1
\end{bmatrix} = \begin{bmatrix}
	2 - \lambda & 1 \\
	1 & 3 - \lambda
\end{bmatrix}
$$

From here we can take the determinant to get the characteristic equation:

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = 0
$$

We compute the determinant and set it equal to zero:

$$
\begin{aligned}
	0 &= \begin{vmatrix}
		2 - \lambda & 1 \\
		1 & 3 - \lambda
	\end{vmatrix} \\[0.75ex]
	0 &=  (2 - \lambda)(3 - \lambda) - (1)(1)  \\[0.75ex]
	0 &=  (6 - 5\lambda + \lambda^2) - 1  \\[0.75ex]
	0 &=  \lambda^2 - 5\lambda + 5
\end{aligned}
$$

We solve the quadratic equation using the quadratic formula to get:

$$
\lambda = \frac{5 \pm \sqrt{(-5)^2 - 4 \cdot 1 \cdot 5}}{2 \cdot 1} = \frac{5 \pm \sqrt{25 - 20}}{2} = \frac{5 \pm \sqrt{5}}{2}
$$

Take the plus and minus result, we find the eigenvalues of the matrix $\mathbf{A}$ to be:

$$
\lambda_1 = \frac{5 + \sqrt{5}}{2} \qquad \lambda_2 = \frac{5 - \sqrt{5}}{2}
$$

{% endcapture %}
{% include example.html content=ex %}

In summary, solving the characteristic equation provides the critical eigenvalues, which in turn reveal fundamental properties of the transformation encoded in $\mathbf{A}$. These properties have profound implications in various physical contexts, including quantum mechanics, stability analysis, and vibrational modes in mechanical systems.












## Physical Interpretations of Eigenvalues

### Quantum Mechanics: Observables and Operators

In quantum mechanics, physical observables (e.g., energy, angular momentum, etc.) can be calculated using mathematical objects called operators. These operators can be represented as matrices. The eigenvalues of these operators correspond to measurable quantities. For example:

- The Hamiltonian $H$ (total energy operator) satisfies $H \psi = E \psi$, where $E$ is the energy eigenvalue, and $\psi$ is the eigenstate.
- The eigenvalues represent possible outcomes of measuring the observable.


Since these eigenvalues represent the measurable physical quantity, they must be purely real. It turns out that this demand, purely real eigenvalues, means these operators/matrices must be **Hermitian**. Let's prove this:


In quantum mechanics, physical observables (e.g., energy, angular momentum, etc.) are calculated using mathematical objects called **operators**. Assuming a finite-dimensional vector space, these operators can be represented as matrices. The eigenvalues of these operators/matrices correspond to the measurable quantities in a quantum system. 

For example, The **Hamiltonian** $\hat{\mathbf{H}}$ (``total" energy operator) satisfies the eigenvalue equation:

$$
\hat{\mathbf{H}} \psi = E \psi,
$$

where $E$ is the energy eigenvalue, and $\psi$ is the corresponding eigenstate (or eigenvector). The eigenvalues $E$ represent the possible energy levels of the system. More generally, the eigenvalues of an operator correspond to the possible outcomes of measuring the observable associated with that operator.

Since the eigenvalues represent measurable physical quantities, they must be **real** numbers. This requirement implies a fundamental property of the operators: they must be **Hermitian**. Let's prove this.

Recall an matrix $\mathbf{H}$ is called **Hermitian** if it satisfies:

$$
\mathbf{H} = \mathbf{H}^\dagger
$$

where $\mathbf{H}^\dagger$ is the conjugate transpose (also called the adjoint) of $\mathbf{H}$. Let $\mathbf{H}$ be a Hermitian operator, and let $\vec{\psi}$ be an eigenvector of $\mathbf{H}$ with eigenvalue $\lambda$, such that:

$$
\mathbf{H} \vec{\psi} = \lambda \vec{\psi}
$$

Acting on the left with the Hermitian transpose of $\vec{\psi}$, we have:

$$
\vec{\psi}^\dagger \mathbf{H} \vec{\psi} = \vec{\psi}^\dagger \lambda \vec{\psi}
$$

$$
\vec{\psi}^\dagger \mathbf{H} \vec{\psi} = \lambda \vec{\psi}^\dagger \vec{\psi}
$$

In quantum mechanics the magnitude of $\vec{\psi}$ is always adjusts to be exactly $1$. This means $\vec{\psi} \cdot \vec{\psi} = \vec{\psi}^\dagger \vec{\psi} = 1^2 = 1$. So, we are left with:

$$
\vec{\psi}^\dagger \mathbf{H} \vec{\psi} = \lambda 
$$

Now, let's take the Hermitian transpose of both sides:

$$
\left( \vec{\psi}^\dagger \mathbf{H} \vec{\psi}\right)^\dagger = \lambda^\dagger 
$$

Recall the taking the transpose of a collection of objects means we have the flip the order of the objects and the the transpose of each object. The same exact thing happens when taking the Hermitian transpose, so we have:

$$
\vec{\psi}^\dagger  \mathbf{H}^\dagger \left( \vec{\psi}^\dagger\right)^\dagger  = \lambda^* 
$$

where we have used the fast that the complex transpose of a scalar is just the complex conjugate. Since $\mathbf{H}$ is Hermitian, that $\mathbf{H}^\dagger = \mathbf{H}$, and taking the Hermitian transpose twice simply returns the original object $ \left( \vec{\psi}^\dagger\right)^\dagger = \vec{\psi}$. This leaves us with:

$$
\vec{\psi}^\dagger  \mathbf{H} \vec{\psi} = \lambda^* 
$$

But, we know that $\mathbf{H} \vec{\psi} = \lambda \vec{\phi} $,

$$
\vec{\psi}^\dagger  \lambda \vec{\phi} = \lambda^* \implies \lambda  \vec{\psi}^\dagger  \vec{\phi} = \lambda^* \implies \lambda = \lambda^*
$$

This implied $\lambda$ is equal to its own complex conjugate. This is only possible if $\lambda$ is real.

So, the eigenvalues of a Hermitian Matrix are real values! 



{% capture ex %}

Consider a spin-1/2 particle (such as an electron) placed in an external magnetic field $\mathbf{B}$ along the $x$-axis. The Hamiltonian for such a system is given by the interaction between the magnetic moment of the particle and the magnetic field:

$$
\mathbf{H} = -\gamma \mathbf{B} \cdot \mathbf{S}
$$

Assume that the magnetic field is aligned along the $x$-axis, i.e., $\mathbf{B} = B_x \hat{x}$, so the Hamiltonian simplifies to:

$$
\mathbf{H} = -\gamma B_x \mathbf{S_x}
$$

where $\gamma$ is the gyromagnetic ratio of the particle in the external magnetic field and relates the magnetic moment of a particle to its angular momentum (or spin). For an electron the gyromagnetic ratio is $\gamma_e = -1.76\times10^{11}$ rad/s/T.  

The $x$-component of the spin of a spin-1/2 particle, $S_x$, has the following matrix representation:

$$
\mathbf{S_x} = \frac{\hbar}{2} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$
Thus, the Hamiltonian matrix becomes:
$$
\mathbf{H} = -\gamma B_x \frac{\hbar}{2} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & -\gamma B_x \frac{\hbar}{2} \\ -\gamma B_x \frac{\hbar}{2} & 0 \end{bmatrix}
$$

We can find energies associated with this problem by setting up the eigenvalue problem:

$$
\mathbf{H} \vec{\psi} = E \vec{\psi}
$$

where $E$ represents the energy eigenvalue, and $\vec{\psi}$ is the corresponding eigenvector. This comes from the fact that when the Hamiltonian operators acts on a state, like $\vec{\psi}$, the it extracts the energy of that state. This gives the following characteristic equation:

$$
\det(\mathbf{H} - E \mathbf{I}) =  \begin{vmatrix} -E & -\gamma B_x \frac{\hbar}{2} \\ -\gamma B_x \frac{\hbar}{2} & -E \end{vmatrix} =  (-E)(-E) - \left( -\gamma B_x \frac{\hbar}{2} \right)^2 = 0
$$

Simplifying and solving for $E$ gives:

$$
E^2 - \left(\gamma B_x \frac{\hbar}{2}\right)^2 = 0 \implies \boxed{E = \pm \gamma B_x \frac{\hbar}{2}}
$$

These two eigenvalues correspond to the possible energy levels of the spin-1/2 particle in the magnetic field along the $x$-axis:

$$
E_1 = -\gamma B_x \frac{\hbar}{2} \qquad\qquad  E_2 = \gamma B_x \frac{\hbar}{2}
$$

These represent the energy of the system in the two possible spin states:

- $E_1$ corresponds to the energy when the spin is aligned opposite to the magnetic field (spin down)
- $E_2$ corresponds to the energy when the spin is aligned with the magnetic field (spin up)

Since the Hamiltonian is off-diagonal, the eigenstates are superposition of the spin up and down states, $\left|\uparrow\right\rangle$ and $\left|\downarrow\right\rangle$, along the $z$-axis. We will see how to find these eigenstates in the next lecture. 

{% endcapture %}
{% include example.html content=ex %}









### Vibrational Modes in Classical Mechanics

In mechanical systems like coupled oscillators or vibrating strings, the eigenvalues correspond to **frequencies** of normal modes, and the eigenvectors describe the mode shapes. The equation:

$$ M \mathbf{\ddot{x}} = K \mathbf{x} \quad \Rightarrow \quad (K - \omega^2 M)\mathbf{x} = 0 $$

gives eigenvalues $\omega^2$ (squared angular frequencies) and eigenvectors $\mathbf{x}$ (mode shapes).


{% capture ex %}

Consider a system of two masses, $m_1$ and $m_2$, connected by three springs with spring constants $k_1$, $k_2$, and $k_3$. The masses are constrained to move along a straight line, and the displacements of $m_1$ and $m_2$ from their equilibrium positions are denoted as $x_1$ and $x_2$, respectively.

<img
src="{{ '/courses/math-methods/images/lec08/SpringsandMasses.png' | relative_url }}"
alt="The image shows a horizontal system made of two blocks connected by three springs. On the far left is a fixed wall attached to the first spring, which pulls on the first block labeled m one. A second spring connects the first block to the second block labeled m two, and a third spring connects the second block to another fixed wall on the far right. Each spring is drawn with a coiled shape, and the spring constants are labeled k one, k two, and k three above them. Beneath each block is a short vertical marker showing the equilibrium position for that block, with a horizontal arrow pointing to the right and labeled x one under the first block and x two under the second block. This indicates that each block can move horizontally from its own equilibrium position."
style="display:block; margin:1.5rem auto; max-width:400px; width:30%;">

The forces on the masses arise from the restoring forces of the springs, leading to the equations of motion:

$$
m_1 \ddot{x}_1 = -k_1 x_1 + k_2 (x_2 - x_1) \implies m_1 \ddot{x}_1 = -(k_1+k_2) x_1 + k_2 x_2 
$$

$$
m_2 \ddot{x}_2 = -k_3 x_2 + k_2 (x_1 - x_2) \implies m_2 \ddot{x}_2 = k_2 x_1 -(k_2 + k_3) x_2
$$

We write this equation in matrix form as:

$$
\mathbf{M} \ddot{\vec{x}} = -\mathbf{K} \vec{x}
$$

where:

$$
\vec{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \quad
\ddot{\vec{x}} = \begin{bmatrix} \ddot{x}_1 \\ \ddot{x}_2 \end{bmatrix} \quad
\mathbf{M} = \begin{bmatrix} m_1 & 0 \\ 0 & m_2 \end{bmatrix} \quad
\mathbf{K} = \begin{bmatrix} k_1 + k_2 & -k_2 \\ -k_2 & k_2 + k_3 \end{bmatrix}
$$

Assuming solutions of the form:

$$
\vec{x}(t) = \vec{v} e^{i \omega t}
$$

where $\omega$ is the angular frequency, the equations of motion reduce to the eigenvalue problem:

$$
\mathbf{M} (-\omega^2)\vec{v} \,\,\, \cancel{e^{i \omega t}}  = -\mathbf{K} \vec{v} \,\,\, \cancel{e^{i \omega t}} \implies \left( \mathbf{K} - \omega^2 \mathbf{M} \right) \vec{v} = 0.
$$

For simplicity let's assume $m_1 = m_2 = m$ and $k_1 = k_2 = k_3 = k$. The matrices simplify to:

$$
\mathbf{M} = m \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \qquad
\mathbf{K} = k \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}
$$

Putting these into the eigenvalue problem equation and dividing through by $m$, the eigenvalue problem gives us:

$$
\left( \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} - \omega^2 \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) \mathbf{v} = 0,
$$

or equivalently:

$$
\begin{bmatrix} 2 - \omega^2 & -1 \\ -1 & 2 - \omega^2 \end{bmatrix} \mathbf{v} = 0.
$$

To find the eigenvalues we can find the characteristic equation and solve for the eigenvalues:

$$
\begin{vmatrix} 2 - \omega^2 & -1 \\ -1 & 2 - \omega^2 \end{vmatrix}  = (2 - \omega^2)^2 - (-1)^2 = 0 \quad\implies\quad (2 - \omega^2)^2 = 1 \quad\implies\quad 2 - \omega^2 = \pm 1
$$

This gives us:

$$
\omega^2 = 1 \qquad \omega^2 = 3
$$

From this process we can determine that the angular frequencies for this problem are:

$$
\omega_1 = 1 \quad \text{and} \quad \omega_2 = \sqrt{3}
$$

The eigenvalues $\omega_1^2 = 1$ and $\omega_2^2 = 3$ correspond to the squares of the vibrational frequencies of the system. 

In the **lower-frequency mode** ($\omega_1$): This frequency is associated with both masses oscillating in phase, meaning they move in the same direction at the same time.

In the **higher-frequency mode** ($\omega_2$): This frequency is associated with both masses oscillate out of phase, meaning one mass moves left while the other moves right.

We will see how to get the eigenvectors, which determines if the masses are oscillating in or out of phase.

{% endcapture %}
{% include example.html content=ex %}











## Application:

Consider a 2D incompressible fluid with a linear shear velocity profile:

$$
u(y) = Uy
$$

where $ U $ is a constant. The velocity gradient tensor (a matrix) for this flow can be approximately represented by the matrix:

$$
\mathbf{A} = \begin{bmatrix}
	0 & U \\
	0 & 0
\end{bmatrix}
$$

The eigenvalues $\lambda$ of $\mathbf{A}$ are found by solving the characteristic equation:

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = 0
$$

where $\mathbf{I}$ is the identity matrix. This gives:

$$
\det(\mathbf{A} - \lambda \mathbf{I}) =
\begin{vmatrix}
	- \lambda & U \\
	0 & - \lambda
\end{vmatrix} = (-\lambda)(-\lambda) - (0)(U) = \lambda^2
$$

Thus, the eigenvalues are:

$$
\lambda = 0
$$

a repeated eigenvalue. We will discuss the meaning of this next lecture where eigenvectors help shed light on the mathematical interpretation of these situations. In this case, we can simply say that there is a degeneracy in the eigenvalue (that is, the eigenvalue is repeated).

Physically, the eigenvalues of the velocity gradient tensor describe the local stability of the flow. In this case, the eigenvalue $\lambda = 0$ indicates that flow perturbations neither grow nor decay exponentially in time. Instead, this reflects a **neutral stability** for this idealized linear shear flow. 




## Problem:


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
\end{bmatrix}
$$

The stability of rotation about a principal axis is determined by the eigenvalues of the following matrix:

$$
\mathbf{A} = \begin{bmatrix}
	0 & \frac{I_3 - I_1}{I_1} \omega_3 & \frac{I_2 - I_1}{I_1} \omega_2 \\
	\frac{I_3 - I_2}{I_2} \omega_3 & 0 & \frac{I_1 - I_2}{I_2} \omega_1 \\
	\frac{I_2 - I_3}{I_3} \omega_2 & \frac{I_1 - I_3}{I_3} \omega_1 & 0
\end{bmatrix}
$$



a)   For $I_1 = 2$, $I_2 = 3$, $I_3 = 5$, and $\omega_1 = 1$, $\omega_2 = 2$, $\omega_3 = 3$, find $\mathbf{A}$ for these numbers.  
b) Set up the characteristic equation for matrix $\mathbf{A}$.   
c) Find the eigenvalues for matrix $\mathbf{A}$. 




















