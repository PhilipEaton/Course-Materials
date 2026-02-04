---
layout: default
title: Mathematical Methods - Lecture 08
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 8
---



# Lecture 08 – The Eigenvalue Problem and Eigenvalues

## Eigenvalue Problem

In this lecture, we explore one of the most important problems in linear algebra when it comes to applications in physics: the **eigenvalue problem**. This arises when a square matrix $\mathbf{A}$ acts on a vector $\vec{v}$ and returns the same vector, scaled by a factor $\lambda$:

$$ \mathbf{A} \vec{v} = \lambda\ \vec{v} $$

It bears repeating what this equation tells us: applying matrix $\mathbf{A}$ to vector $\vec{v}$ results in the same vector, just scaled by a factor $\lambda$. The direction remains unchanged—except in the case where $\lambda$ is negative, in which case the vector flips direction.

Why should we, as physicists, care? Simply put, the eigenvalue problem appears in many foundational equations and physical systems, from quantum mechanics and classical mechanics to the stability analysis of dynamic systems.






## Mathematical Framework

Let’s begin by exploring the mathematical definitions of eigenvalues and eigenvectors. We’ll later connect them to physical interpretations, since those depend on the context of the specific problem being solved.

### Definition of Eigenvalues and Eigenvectors

Given a square ($n \times n$) matrix $\mathbf{A}$, a nonzero vector $\vec{v}_i$ is called an **eigenvector** (specifically the $i$-th eigenvector) of the matrix if it satisfies:

$$ \mathbf{A} \vec{v} = \lambda_i\ \vec{v}_i $$

where $\lambda_i$ is a scalar called the **eigenvalue** associated with $\vec{v}_i$. Every eigenvector has one, and only one, corresponding eigenvalue. That is,
- eigenvector $\vec{v}_1$ has an associated eigenvalue of $\lambda_1$,
- eigenvector $\vec{v}_2$ has an associated eigenvalue of $\lambda_2$,
- $\qquad\vdots$
- eigenvector $\vec{v}_n$ has an associated eigenvalue of $\lambda_n$.  

For the sake of clarity, we will drop the $i$ subscript until it becomes important to explicity show which eigenvalue belongs to which eigenvector.


If $\mathbf{A}$ is an $n \times n$ matrix, you typically obtain $n$ *unique* eigenvalues, though not always. Sometimes eigenvalues are repeated, we call these eigenvalues **degenerate**. In degenerate eigenvalue cases, the corresponding eigenvectors form a set of unique, linearly independent vectors. For example, if an eigenvalue appears three times, it will have three linearly independent eigenvectors associated with it.

#### Eigenvalues $\ne 0$

One important feature of the eigenvalue problem is that the direction defined by the eigenvector $\vec{v}$ is not rotated by the matrix $\mathbf{A}$, though they can be flipped if their associated eigenvalue is negative. Assuming $\lambda \ne 0$, the eigenvector represents an axis of "rotation" along which the matrix scales objects. Specifically:

- If $\lambda > 0$, $\vec{v}$ is scaled and its direction remains the same.
	- For example, see the red vector below, where the solid blue vector is the original vector before being acted on by $\mathbf{A}$.  
- If $\lambda < 0$, $\vec{v}$ is scaled and  is direction is flipped.
	- For example, see the dashed purple vector below, where the blue vector is the original vector before being acted on by $\mathbf{A}$.  

<img
  src="{{ '/courses/math-methods/images/lec08/Scaling1.png' | relative_url }}"
  alt="The image shows an x–y coordinate plane with the horizontal axis labeled x and the vertical axis labeled y. A blue arrow representing an original vector extends from the origin into the first quadrant. A red arrow, labeled as scaled, points in the same direction as the blue vector but is longer, showing the effect of multiplying the vector by a positive number. Red text near this arrow explains that applying a matrix to the vector with a positive scaling value makes the vector longer in the same direction. A dashed red arrow extends from the origin into the third quadrant, pointing in the opposite direction from the original vector. This arrow is labeled reflected and scaled, with red text explaining that scaling by a negative number reverses the direction and changes the length."
  style="display:block; margin:1.5rem auto; max-width:400px; width:45%;">

#### Eigenvalues $= 0$

If one or more eigenvalue happens to be zero, that means the transformation represented by $\mathbf{A}$ will take the associated eigenvector $\vec{v}$ to the zero vector:

$$ \mathbf{A} \vec{v} = \lambda \vec{v} = 0 \cdot \vec{v} = \vec{0} $$

To be clear, this does not mean that $\vec{v}$ is the zero vector, just that the matrix operation brings it to the zero vector.

In this case, $\vec{v}$ lies in the **null space** (also called the **kernel**) of $\mathbf{A}$. This means $\mathbf{A}$ collapses any vector—or any component of a vector—pointing in this direction to the origin. That is, $\mathbf{A}$ annihilates it. This kind of dimensional collapse can reduce a 3D system to a 2D one, for example. Zero eigenvalues typically arise when $\mathbf{A}$ is a singular matrix. And as we’ve seen before, singular matrices do not have inverses.

<br>

Whether an eigenvalue is positive, negative, or zero gives you powerful geometric insight into what the transformation $\mathbf{A}$ is doing. In physics, these properties help us understand stretching, compression, reflection, or elimination of possible states in systems like quantum mechanics, vibrational modes, and dynamical stability analysis.

Now that we have a conceptual feel for eigenvalues and eigenvectors, let's learn how to find them. In this lecture we will focus on finding eigenvalues. In the next lecture we will extend to finding eigenvectors. 









### Finding Eigenvalues

To determine the eigenvalues $\lambda$ of a square matrix $\mathbf{A}$, we begin by rewriting the eigenvalue equation:

$$ \mathbf{A} \vec{v} = \lambda \vec{v} $$

in the following form:

$$ (\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \vec{0} $$

where $\mathbf{I}$ is the identity matrix of the same dimension as $\mathbf{A}$, $\vec{v}$ is the eigenvector associated with the eigenvalue $\lambda$, and $\vec{0}$ is the zero vector.

We are looking for a nontrivial solution (that is, $\vec{v} \neq \vec{0}$), because the trivial solution $\vec{v} = \vec{0}$ implies all the eigenvectors are zero, which is boring. 

To find such solutions, suppose for a moment the matrix $(\mathbf{A} - \lambda \mathbf{I})$ *does* have an inverse. Then we could write:

$$
\begin{aligned}
(\mathbf{A} - \lambda \mathbf{I}) \vec{v} &= \vec{0} \\[0.75ex]
(\mathbf{A} - \lambda \mathbf{I})^{-1}(\mathbf{A} - \lambda \mathbf{I}) \vec{v} &= (\mathbf{A} - \lambda \mathbf{I})^{-1}\vec{0} \\[0.75ex]
\mathbf{I}\vec{v} &= \vec{0} \\[0.75ex]
\vec{v} &= \vec{0}
\end{aligned}
$$

But this result contradicts our requirement that $\vec{v} \neq \vec{0}$. In mathematics, when we reach a contradiction, it means one of our assumptions must be incorrect. So far we assumed:

1. A nontrivial solution $\vec{v} \neq \vec{0}$ exists.
2. The matrix $(\mathbf{A} - \lambda \mathbf{I})$ has an inverse.

One of these must be wrong! We know we want to find nontrivial solutions. This means the assumption that the matrix $(\mathbf{A} - \lambda \mathbf{I})$ has an inverse is wrong. So, $(\mathbf{A} - \lambda \mathbf{I})$ **does not have an inverse**. 

Recall, a matrix that does not have an inverse is called a **singular matrix**, which means its determinant ia zero. This gives us a condition to find our eigenvalues:

{% capture ex %}

To obtain nontrivial solutions ($\vec{v} \neq \vec{0}$) to the eigenvalue problem:

$$ \mathbf{A} \vec{v} = \lambda \vec{v} \implies (\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \vec{0} $$

we require:

$$ \det(\mathbf{A} - \lambda \mathbf{I}) = 0 $$

This equation is called the **characteristic equation**, and its solutions give the eigenvalues $\lambda$ of the matrix $\mathbf{A}$.

{% endcapture %}
{% include result.html content=ex %}

For an $n \times n$ matrix $\mathbf{A}$, the characteristic equation is a polynomial of degree $n$. This means there can be up to $n$ eigenvalues, which may be real or complex. These eigenvalues can be distinct or repeated (degenerate), with their multiplicities determined by the roots of the polynomial.

Let’s work through an example.

{% capture ex %}

Consider the matrix:

$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix}
$$

We want to find the eigenvalues $\lambda$ such that:

$$
\mathbf{A} \vec{v} = \lambda\ \vec{v}
$$

has nontrivial solutions. We begin by forming:

$$
(\mathbf{A} - \lambda \mathbf{I}) \vec{v} = \vec{0}
$$

where $\mathbf{I}$ is the $2 \times 2$ identity matrix. This gives the matrix:

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

Next, we compute the determinant and set it equal to zero to get the characteristis equation:

$$
\begin{aligned}
\det(\mathbf{A} - \lambda \mathbf{I}) &= 0\\
\begin{vmatrix}
2 - \lambda & 1 \\
1 & 3 - \lambda
\end{vmatrix} &= 0 \\[0.75ex]
(2 - \lambda)(3 - \lambda) - (1)(1) &= 0 \\[0.75ex]
(6 - 5\lambda + \lambda^2) - 1 &= 0 \\[0.75ex]
\lambda^2 - 5\lambda + 5 &= 0
\end{aligned}
$$

This is a quadratic polynomial, as expected for a $2 \times 2$ matrix.

We now solve using the quadratic formula:

$$
\lambda = \frac{5 \pm \sqrt{(-5)^2 - 4(1)(5)}}{2(1)}
= \frac{5 \pm \sqrt{25 - 20}}{2}
= \frac{5 \pm \sqrt{5}}{2}
$$

Thus, the eigenvalues of $\mathbf{A}$ are

$$
\lambda_1 = \frac{5 + \sqrt{5}}{2}
\qquad\qquad
\lambda_2 = \frac{5 - \sqrt{5}}{2}
$$

{% endcapture %}
{% include example.html content=ex %}

In summary, solving the characteristic equation yields the eigenvalues of $\mathbf{A}$, which reveal key geometric and physical properties of the transformation. These ideas play a central role in physics, including quantum mechanics, stability analysis, and vibrational motion. Let’s now examine some of those physical interpretations.








## Applications

### Quantum Mechanics: Observables and Operators

In quantum mechanics, physical observables (e.g., energy, angular momentum, etc.) can be calculated using mathematical objects called operators. Operators, as we have often discussed, can be represented as matrices. The eigenvalues of these operators correspond to measurable quantities. 

For example, the **Hamiltonian** $\mathbf{H}$ ("total" energy operator) satisfies the eigenvalue equation:

$$
\mathbf{H} \psi = E\ \psi
$$

where $E$ is the energy eigenvalue, and $\psi$ is the corresponding eigenstate (or eigenvector). The eigenvalues $E$ represent the possible energy levels of the system. 

More generally, the eigenvalues of an operator correspond to the possible outcomes of measuring the observable associated with that operator. Since these eigenvalues represent measurable physical quantities, they must be purely real. The demand of purely real eigenvalues for an operator/matrix means the operator/matrix must be **Hermitian**. Let's prove this.

Recall a matrix $\mathbf{H}$ is **Hermitian** if it is equal to its own adjoint:

$$
\mathbf{H} = \mathbf{H}^\dagger
$$

where $\mathbf{H}^\dagger$ is the adjoint of $\mathbf{H}$. 

Let $\mathbf{H}$ be a Hermitian operator/matrix, and let $\vec{\psi}$ be an eigenvector of $\mathbf{H}$ with an eigenvalue $\lambda$, such that:

$$
\mathbf{H} \vec{\psi} = \lambda\ \vec{\psi}
$$

Since the matrix in question could be complex, otherwise taking the adjoint over the transpose makes no sense, we must assume the eignevector can also be complex. This means instead of taking the transpose of $\vec{\psi}$, we need to take its adjoint to account for its potentially complex nature. 

Now, acting on the eigenvalue problem on the left with $\vec{\psi}^\dagger$, we have:

$$
\vec{\psi}^\dagger \mathbf{H} \vec{\psi} = \vec{\psi}^\dagger \lambda \vec{\psi}
$$

$$
\vec{\psi}^\dagger \mathbf{H} \vec{\psi} = \lambda \vec{\psi}^\dagger \vec{\psi}
$$

In quantum mechanics the magnitude of $\vec{\psi}$ is always adjusted to be exactly $1$. This means $\vec{\psi} \cdot \vec{\psi} = \vec{\psi}^\dagger \vec{\psi} = 1^2 = 1$. So, we are left with:

$$
\vec{\psi}^\dagger \mathbf{H} \vec{\psi} = \lambda 
$$

Now, let's take the adjoint of both sides:

$$
\left( \vec{\psi}^\dagger \mathbf{H} \vec{\psi}\right)^\dagger = \lambda^\dagger 
$$

Recall the taking the transpose of a collection of objects means we have the flip the order of the objects and take transpose of each object. The same exact thing happens when taking the adjoint. We have:

$$
\vec{\psi}^\dagger  \mathbf{H}^\dagger \left( \vec{\psi}^\dagger\right)^\dagger  = \lambda^* 
$$

where we have used the fact that the adjoint of a scalar is just the complex conjugate. 

We can make a couple of simplifications at this point, 
1) $\mathbf{H}$ is Hermitian, meaning that $\mathbf{H}^\dagger = \mathbf{H}$, and 
2) taking the adjoint of an adjoint the original object $ \left( \vec{\psi}^\dagger\right)^\dagger = \vec{\psi}$. 

This leaves us with:

$$
\vec{\psi}^\dagger  \mathbf{H} \vec{\psi} = \lambda^* 
$$

But, we know from the original eigenvalue problem that $\mathbf{H} \vec{\psi} = \lambda \vec{\psi} $. Putting this in gives us:

$$
\vec{\psi}^\dagger  \lambda \vec{\psi} = \lambda^* \quad\implies\quad \lambda  \vec{\psi}^\dagger  \vec{\psi} = \lambda^* \quad\implies\quad \lambda = \lambda^*
$$

For $\lambda$ to be the eigenvalue to the eigenvalue problem for a Hermitian operator/matrix, it must be that $\lambda$ is equal to its own complex conjugate. This is only possible if $\lambda$ is **real**.

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

where $\gamma$, called the gyromagnetic ratio, relates the magnetic moment of a particle to its angular momentum (or spin). For an electron the gyromagnetic ratio is $\gamma_e = -1.76\times10^{11}$ rad/s/T.  

The $x$-component of the spin of a spin-1/2 particle, $S_x$, has the following matrix representation:

$$
\mathbf{S_x} = \frac{\hbar}{2} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

You may recognize this matrix from our discussion of quaternions. Though, there is some minor reassignments that have been done, along with an extra complex units $i$, to make the $z$-direction the spin-up/down direction, as is convention.

Anyways, using this matrix the Hamiltonian matrix becomes:

$$
\mathbf{H} = -\gamma B_x \frac{\hbar}{2} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & -\gamma B_x \frac{\hbar}{2} \\ -\gamma B_x \frac{\hbar}{2} & 0 \end{bmatrix}
$$

We can find energies associated with this problem by setting up the eigenvalue problem:

$$
\mathbf{H} \vec{\psi} = E \vec{\psi}
$$

where $E$ represents the energy eigenvalue, and $\vec{\psi}$ is the corresponding eigenvector. This comes from the definition of the Hamiltonian operator: when the Hamiltonian operator acts on a state, like $\vec{\psi}$, it extracts the energy of that state. 

We can find these eigen-energies by finding and solving the characteristic equation:

$$
\det(\mathbf{H} - E \mathbf{I}) =  \begin{vmatrix} -E & -\gamma B_x \frac{\hbar}{2} \\ -\gamma B_x \frac{\hbar}{2} & -E \end{vmatrix} =  (-E)(-E) - \left( -\gamma B_x \frac{\hbar}{2} \right)^2 = 0
$$

Simplifying and solving for $E$ gives:

$$
E^2 - \left(\gamma B_x \frac{\hbar}{2}\right)^2 = 0 \implies \boxed{E = \pm \gamma B_x \frac{\hbar}{2}}
$$

These two eigenvalues correspond to the possible energy levels of the spin-1/2 particle in the magnetic field along the $x$-axis:

$$
E_1 = -\gamma B_x \frac{\hbar}{2} \qquad\qquad  E_2 = +\gamma B_x \frac{\hbar}{2}
$$

These represent the energy of the system in the two possible spin states:

- $E_1$ corresponds to the energy when the spin is aligned with the magnetic field (spin up)
- $E_2$ corresponds to the energy when the spin is aligned opposite to the magnetic field (spin down)

Since the Hamiltonian is off-diagonal, the eigenstates are superposition of the spin up and down states, $\\lvert\uparrow\right\rangle$ and $\lvert\downarrow\right\rangle$, along the $z$-axis. We will see how to find these eigenstates in the next lecture. 

{% endcapture %}
{% include example.html content=ex %}









### Vibrational Modes in Classical Mechanics

In mechanical systems like coupled oscillators or vibrating strings, the eigenvalues correspond to **frequencies** of normal modes, and the eigenvectors describe the mode shapes. 

The general equation for a system of connected simple harmonic oecillators can be written as:

$$ \mathbf{M} \ddot{\vec{x}} = -\mathbf{K} \vec{x} $$

where $\mathbf{M}$ is the inertia matrix, $\mathbf{K}$ is the spring constant matrix, and $\vec{x}$ is the position vector of the center of mass of each object in the system. To get this into eigenvalue problem form, we assume the positions of the masses will all oscillate with the same frequency $\omega$. This implies:

$$ 
x_i(t) \propto \sin(\omega t) \quad\Rightarrow\quad \dot{x}_i(t) \propto \omega \cos(\omega t) \quad\Rightarrow\quad \ddot{x}_i(t) \propto -\omega^2 \sin(\omega t) \quad\Rightarrow\quad \ddot{x}_i(t) \propto -\omega^2 x_i(t)
$$

This assumption leaves us with:

$$ \omega^2 \mathbf{M} \vec{x} = \mathbf{K} \vec{x} $$

which could be rearranged to get:

$$ \mathbf{M}^{-1} \mathbf{K} \vec{x} = \omega^2  \vec{x} $$

an eigenvalue problem with eigenvalues representing the square of the oscllation frequencies all of the masses could have $\omega^2$. These shared frequencies are called the normal mode frequencies, and the associated eigenvectors are called the normal modes for the system. These are vital to the student of many body systems and oscillations in crystaline structures. 

To solve for the frequiencies we can rearrange this into equal zero form:

$$  \left(\mathbf{M}^{-1} \mathbf{K} - \omega^2 \mathbf{I} \right) \vec{x} = \vec{0}  $$

where we can multiple on the left by $\mathbf{M}$ to get:

$$  \left(\mathbf{K} - \omega^2 \mathbf{M} \right) \vec{x} = \vec{0} $$

This helps us avoid having to take the inverse of the inertia matrix $\mathbf{M}$, which is nice since inverses are often annoying to calculate.

The characteristic equation, in this case, will be given by:

$$  \det\left(\mathbf{K} - \omega^2 \mathbf{M} \right) = 0 $$

whcih can be used to find the eigenvalues $\omega^2$ (squared angular frequencies).


{% capture ex %}

Consider a system of two masses, $m_1$ and $m_2$, connected by three springs with spring constants $k_1$, $k_2$, and $k_3$. The masses are constrained to move along a straight line, and the displacements of $m_1$ and $m_2$ from their equilibrium positions are denoted as $x_1$ and $x_2$, respectively.

<img
src="{{ '/courses/math-methods/images/lec08/SpringsandMasses.png' | relative_url }}"
alt="The image shows a horizontal system made of two blocks connected by three springs. On the far left is a fixed wall attached to the first spring, which pulls on the first block labeled m one. A second spring connects the first block to the second block labeled m two, and a third spring connects the second block to another fixed wall on the far right. Each spring is drawn with a coiled shape, and the spring constants are labeled k one, k two, and k three above them. Beneath each block is a short vertical marker showing the equilibrium position for that block, with a horizontal arrow pointing to the right and labeled x one under the first block and x two under the second block. This indicates that each block can move horizontally from its own equilibrium position."
style="display:block; margin:1.5rem auto; max-width:400px; width:45%;">

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
x_i(t) \propto \sin(\omega t) \quad\Rightarrow\quad x_i(t) \propto -\omega^2 x_i(t)
$$

allows us to rewrite in the following manner:

$$
\ddot{\vec{x}} = \begin{bmatrix} \ddot{x}_1 \\ \ddot{x}_2 \end{bmatrix} = -\omega^2 \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}  = -\omega^2\vec{x}
$$

where $\omega$ is the angular frequency. Putting this into the above matrix equation gives us the eigenvalue problem:

$$
\omega^2\mathbf{M} \vec{x} = \mathbf{K} \vec{x} \quad\Rightarrow\quad \left( \mathbf{K} - \omega^2 \mathbf{M} \right) \vec{x} = 0
$$

For simplicity let's assume $m_1 = m_2 = m$ and $k_1 = k_2 = k_3 = k$. The matrices simplify to:

$$
\mathbf{M} = m \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \qquad\qquad \mathbf{K} = k \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}
$$

Putting these into the eigenvalue problem equation and dividing through by $m$, the eigenvalue problem gives us:

$$
\left( \omega_0^2 \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} - \omega^2 \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) \vec{x} = 0
$$

where $\omega_0^2 = \frac{k}{m}$. we can combine these matrices to get:

$$
\begin{bmatrix} 2\omega_0^2 - \omega^2 & -\omega_0^2 \\ -\omega_0^2 & 2\omega_0^2 - \omega^2 \end{bmatrix} \vec{x} = 0
$$

To find the eigenvalues we can set up the characteristic equation and solve:

$$
\begin{aligned}
\begin{vmatrix} 2\omega_0^2 - \omega^2 & -\omega_0^2 \\ -\omega_0^2 & 2\omega_0^2 - \omega^2 \end{vmatrix} \vec{x} &= 0 \\[3ex]
(2\omega_0^2 - \omega^2)^2 - (-\omega_0^2)^2 &= 0 \\[1.15ex]
(2\omega_0^2 - \omega^2)^2 &= (\omega_0^2)^2 \\[1.15ex]
2\omega_0^2 - \omega^2 &= \pm\omega_0^2 \\[1.15ex]
-\omega^2 &= (\pm 1 - 2) \omega_0^2 \\[1.15ex]
 \omega^2 &= -(\pm 1 - 2) \omega_0^2 \\[1.15ex]
 \omega^2 &= ( 2 \pm 1) \omega_0^2 \\[1.15ex]
\end{aligned}
$$

This gives us:

$$
\omega^2 = \omega_0^2 \qquad \omega^2 = 3\omega_0^2
$$

From this process we can determine that the angular frequencies for this problem are:

$$
\omega_1 = \omega_0 \quad \text{and} \quad \omega_2 = \sqrt{3}\ \omega_0^2
$$

If you were to find the associated eigenvectors to this problem, you would find that:
- The **lower-frequency mode** ($\omega_1$)is associated with both masses oscillating in phase.
	- The masses move in the same direction at the same time.
- The **higher-frequency mode** ($\omega_2$) is associated with both masses oscillate out of phase.
	- One mass moves left while the other moves right, and vice versa.

Again, we will see how to find eigenvectors in the next lecture.

{% endcapture %}
{% include example.html content=ex %}












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



a) For $I_1 = 2$, $I_2 = 3$, $I_3 = 5$, and $\omega_1 = 1$, $\omega_2 = 2$, $\omega_3 = 3$, find $\mathbf{A}$ for these numbers.  
b) Set up the characteristic equation for matrix $\mathbf{A}$.   
c) Find the eigenvalues for matrix $\mathbf{A}$. 




















