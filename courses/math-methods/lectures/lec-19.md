---
layout: default
title: Mathematical Methods - Lecture 19
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 19
---


# Lecture 19 – Systems of Ordinary Differential Equations


## Motivation: Why Study Systems of ODEs?

Up until now, we've only worked with models represented by a single differential equation. For example, only one differential equation was needed to model the motion of a simple harmonic oscillator with damping, the ODE being given by an application of Newton's Second Law and force analysis. That is, we've been solving equations of the form:

$$
\frac{dy}{dt} = f(y,t)
$$

where we track a *single* unknown function, $ y(t) $, over time. However, real-world systems often involve multiple interacting components, leading to **coupled** differential equations—where two or more unknown functions evolve together. This is where **systems of ODEs** come into play. 

In the context of coupled ODEs, the word ``coupled" can be interpreted as ``connected." The connection typically arises because the different variables in the system interact in some way—meaning the rate of change of one variable depends on the value of another. Often, the strength of this connection is governed by some physical constant, which we call the coupling constant. The coupling constant determines how strongly the variables influence each other. You will hear this term used anywhere coupled ODEs or coupled systems are being analyzed, from masses connected by springs to quantum fields representing protons and electrons coupling together in quantum field theory.

Many physical, biological, and engineering systems are inherently coupled, **multi-variable problems**--or connected, multi-variable problems . These systems describe multiple dependent variables whose mutual connections influence each other as they evolve over time. Some specific examples include:


- **Mechanical systems:** Two masses connected by a spring influence each other’s motion.
- **Electrical circuits:** Current flowing through multiple components interacts through Kirchhoff’s laws.
- **Chemical reactions:** Concentrations of different substances change in response to reaction rates.
- **Population models:** Predator and prey populations depend on each other’s growth rates.
- **Quantum mechanics:** The evolution of quantum states depends on how the state interact with one another if they are not orthogonal.


In all these cases, we need a way to describe the simultaneous evolution of multiple dependent variables. That’s exactly what systems of ODEs allow us to do.

### From One Equation to Many

To illustrate why coupled equations arise naturally, consider a simple example: two masses connected by a spring. In this case the coupling constant will be the spring constant of the spring holding both masses together. 

<img
src="{{ '/courses/math-methods/images/lec19/MassesAttachedByASpring.png' | relative_url }}"
alt="Two rectangular blocks labeled m1 and m2 are connected by a single spring labeled k. The blocks sit horizontally with the spring stretched between them, representing a simple system where the two masses are connected and can move along a line."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">


Each mass moves according to Newton’s Second Law:

$$
m_1 \frac{d^2x_1}{dt^2} = -k(x_1 - x_2)
$$

$$
m_2 \frac{d^2x_2}{dt^2} = -k(x_2 - x_1)
$$

Notice that the acceleration of $ m_1 $ depends on $ x_2 $, and vice versa. This means we **cannot** solve these equations independently. Instead, we have a **coupled system of equations**, where the two functions $ x_1(t) $ and $ x_2(t) $ must be solved together.

The most efficient way to analyze and solve coupled systems like this is through **matrix notation**. To see why, let’s first rewrite the equations in a cleaner form:

$$
\begin{aligned}
	m_1 \frac{d^2x_1}{dt^2} = -k(x_1 - x_2) \\
	m_2 \frac{d^2x_2}{dt^2} = -k(x_2 - x_1)
\end{aligned} \quad\implies\quad
\begin{aligned}
	\frac{d^2x_1}{dt^2} = -\frac{k}{m_1}(x_1 - x_2) \\
	\frac{d^2x_2}{dt^2} = -\frac{k}{m_2}(x_2 - x_1)
\end{aligned} \quad\implies\quad
\begin{aligned}
	\frac{d^2x_1}{dt^2} = - \omega_1 x_1 + \omega_1 x_2 \\
	\frac{d^2x_2}{dt^2} =   \omega_2 x_1 - \omega_2 x_2
\end{aligned}
$$

where we have introduced the angular frequencies $ \omega_1 = \frac{k}{m_1} $ and $ \omega_2 = \frac{k}{m_2} $ to simplify notation. Now, we can express this system more compactly in matrix form.

First, we can rewrite the system in matrix form as: 

$$
\frac{d^2}{dt^2} \begin{bmatrix}
	x_1 \\ x_2
\end{bmatrix} = \begin{bmatrix}
	-\omega_1 & \omega_1 \\
	\omega_2 & - \omega_2
\end{bmatrix} \begin{bmatrix}
	x_1 \\ x_2
\end{bmatrix}
$$

Then, defining the vector:

$$
\vec{x} =
\begin{bmatrix}
	x_1 \\ x_2
\end{bmatrix}
$$

allows us to write the system in a more compact form as:

$$
\frac{d^2}{dt^2} \vec{x} = \begin{bmatrix}
	-\omega_1 & \omega_1 \\
	\omega_2 & - \omega_2
\end{bmatrix} \vec{x}
$$

This formulation highlights the key role of the **coefficient matrix**, which encodes the coupling between the two masses. As we will see, solving this system reduces to understanding the properties of this matrix—specifically, its eigenvalues and eigenvectors.

Before solving this problem, let’s take a step back and consider a slightly more general set up.


## Writing a System of ODEs in Matrix Form

Now that we've seen why and how systems of ODEs naturally arise in physics and engineering, we need a structured way to express them. As we just saw, the best tool for this job is to use matrices and vectors in their compact **matrix notation** form.

To illustrate, let’s consider a simple system of two coupled, first-order ODEs:

$$
\frac{dx_1}{dt} = a_{11}x_1 + a_{12}x_2
$$

$$
\frac{dx_2}{dt} = a_{21}x_1 + a_{22}x_2
$$

Instead of writing these separately, we recognize they share a common structure. If we define:

$$
\vec{x} =
\begin{bmatrix}
	x_1 \\ x_2
\end{bmatrix}
\qquad\qquad
\mathbf{A} =
\begin{bmatrix}
	a_{11} & a_{12} \\
	a_{21} & a_{22}
\end{bmatrix}
$$

then the two individual ODEs can be written as a single matrix ODE of the form:

$$
\frac{d}{dt} \vec{x} = \mathbf{A} \vec{x}
$$

This is the **standard form** of a linear system of ODEs.


{% capture ex %}
Let’s say we have the system:

$$
\frac{dx_1}{dt} = 3x_1 + 4x_2
$$

$$
\frac{dx_2}{dt} = -x_1 + 2x_2
$$

Rewriting in matrix form, we can define:

$$
\vec{x} =
\begin{bmatrix}
	x_1 \\ x_2
\end{bmatrix} \qquad\qquad
\mathbf{A} =
\begin{bmatrix}
	3 & 4 \\
	-1 & 2
\end{bmatrix}
$$

to rewrite the system of ODEs in the following manner:

$$
\frac{d}{dt} \vec{x} =
\begin{bmatrix}
	3 & 4 \\
	-1 & 2
\end{bmatrix}
\vec{x} \quad\implies\quad \frac{d}{dt} \vec{x} = \mathbf{A} \vec{x}
$$

This is much more compact and sets us up to use powerful matrix techniques to solve the system.
{% endcapture %}
{% include example.html content=ex %}





### Higher-Dimensional Systems

The same approach extends naturally to larger systems. If we have $ n $ coupled first-order ODEs:

$$
\frac{dx_i}{dt} = \sum_{j=1}^{n} a_{ij} x_j, \quad \text{for } i = 1,2, \dots, n
$$

we can define:

$$
\vec{x} =
\begin{bmatrix}
	x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
\qquad\qquad
\mathbf{A} =
\begin{bmatrix}
	a_{11} & a_{12} & \dots & a_{1n} \\
	a_{21} & a_{22} & \dots & a_{2n} \\
	\vdots & \vdots & \ddots & \vdots \\
	a_{n1} & a_{n2} & \dots & a_{nn}
\end{bmatrix}
$$

Then the entire system reduces to:

$$
\frac{d}{dt} \vec{x} = \mathbf{A} \vec{x}
$$

Once we have the system in matrix form, we can bring in tools from linear algebra, like eigenvalues and eigenvectors, to analyze and solve it.

### Interpreting the Matrix Equation

What does this equation really mean? The derivative $ \frac{d}{dt} \vec{x} $ tells us how the system evolves over time and the matrix $ \mathbf{A} $ controls that evolution. In a sense, $ \mathbf{A} $ acts as an **operator** that transforms $ \vec{x} $ at any given moment, determining how its components change over time.

A particularly important observation is that the solution structure of this system is dictated by the **eigenvalues and eigenvectors** of $ \mathbf{A} $. If $ \vec{x} $ aligns with an eigenvector of $ \mathbf{A} $, its growth or decay will follow a simple exponential law, as we will prove later. This insight will be crucial when we solve these systems in the next sections.









## Solving Homogeneous Systems Using Eigenvalues and Eigenvectors

With our system of ODEs rewritten in matrix form, let’s figure out how to actually solve it. The key idea is that a system of first-order linear ODEs behaves like a matrix transformation evolving in time, and the most effective way to understand how a matrix behaves is through its **eigenvalues and eigenvectors**.

For now, let's focus on a homogeneous system of linear first-order ODEs, meaning there are no external driving terms:

$$
\frac{d}{dt} \vec{x} = \mathbf{A} \vec{x}
$$

where $ \vec{x} $ is an $ n $-dimensional column vector and $ \mathbf{A} $ is an $ n \times n $ matrix. Our goal is to find an explicit solution for $ \vec{x}(t) $.

### The Eigenvalue Approach

To break this problem down conceptually, we can frame it as addressing two fundamental questions:


1) How can we best satisfy the derivatives and their relationships to the functions?  
2) How can we effectively incorporate the multidimensional nature of the problem?


The first question can be addressed by making a simple, yet powerful observation about exponentials:

$$
\frac{d}{dt} e^{at} = a e^{at}
$$

At first glance, this is just a basic derivative rule. However, rewriting the exponential function in terms of an arbitrary function, $ f(t) = e^{at} $, we see:

$$
\frac{d}{dt} f(t) = a f(t)
$$

This is interesting because the derivative operator acts on $ f(t) $ and returns the same function multiplied by a constant. This mirrors the concept of **eigenvalues and eigenvectors**, except instead of a vector, we have a function acting as an *eigenfunction* of the derivative operator. This suggests that exponentials are natural candidates for solutions to differential equations, and it the main reason they seem to appear in almost every solution to an ODE we have seen thus far.

So, to answer the first question: how to best satisfy the derivative structure of our system, we can **assume the solution has an exponential dependence on the working variable**.

The second question is one we already answered in linear algebra. The best way to handle the multidimensional nature of the problem is to apply an **eigenvalue/eigenvector analysis** to the system. This leads us to an educated guess for the form of the solution:

$$
\vec{x}(t) = \vec{v} e^{\lambda t}
$$

where $ \vec{v} $ is a constant vector, presumably related to the eigenvector, and $ \lambda $ is a scalar, presumably related to the eigenvalue. Plugging this into our system of equations:

$$
\begin{aligned}
	\frac{d}{dt} \vec{x} &= \mathbf{A} \vec{x} \\[1.5ex]
	\frac{d}{dt} \left( \vec{v} e^{\lambda t} \right) &= \mathbf{A} \left( \vec{v} e^{\lambda t} \right) \\[1.5ex]
	\vec{v} \left( \frac{d}{dt}  e^{\lambda t} \right) &= \mathbf{A} \vec{v}  \, e^{\lambda t} \\[1.5ex]
	\vec{v} \left( \lambda  e^{\lambda t} \right) &= \mathbf{A} \vec{v}  \, e^{\lambda t} \\[1.5ex]
	\lambda \vec{v} \, e^{\lambda t} &= \mathbf{A} \vec{v}  \, e^{\lambda t}
\end{aligned}
$$

Since exponentials are never zero, we can cancel $ e^{\lambda t} $ on both sides, leaving:

$$
\mathbf{A} \vec{v} = \lambda \vec{v}
$$

which is precisely the **eigenvalue equation**. This tells us that for each eigenvalue $ \lambda $ of $ \mathbf{A} $, there exists a corresponding eigenvector $ \vec{v} $ and eigenfunction $e^{\lambda t}$, and each such pair produces a solution of the form:

$$
\vec{x}(t) = \vec{v} e^{\lambda t}
$$

This is an incredibly useful result. By finding the eigenvalues and eigenvectors of $ \mathbf{A} $, we can construct the general solution to the system, gaining insight into how each independent mode of the system evolves over time.

If $ \mathbf{A} $ is an $ n \times n $ matrix with $ n $ linearly independent eigenvectors $ \vec{v}_1, \vec{v}_2, \dots, \vec{v}_n $ corresponding to eigenvalues $ \lambda_1, \lambda_2, \dots, \lambda_n $, then the most general solution is a linear combination of these solutions:

$$
\vec{x}(t) = C_1 \vec{v}_1 e^{\lambda_1 t} + C_2 \vec{v}_2 e^{\lambda_2 t} + \dots + C_n \vec{v}_n e^{\lambda_n t}
$$

The constants $ C_1, C_2, \dots, C_n $ are determined by the initial conditions $ \vec{x}(0) $ of the problem. 

Each term in the general solution represents a **mode** of the system’s behavior. The eigenvalues $ \lambda_1, \lambda_2, \dots, \lambda_n $ determine the growth, decay, and/or oscillation frequency of each mode in the following manners:

- If all eigenvalues are negative, solutions decay to zero (stable system).
- If any eigenvalue is positive, at least one mode grows exponentially (unstable system).
- If eigenvalues are complex, the system oscillates.


The eigenvectors $ \vec{v}_1, \vec{v}_2, \dots, \vec{v}_n $ represent the motion of the system within each of the modes. The specifics of these vectors will depend on how you parameterize the system. 


{% capture ex %}
Let’s work through a simple example. Consider the system:

$$
\begin{array}{l}
	\frac{d x_1}{dt} = 4 x_1 - 2 x_2 \\
	\frac{d x_2}{dt} = x_1 + x_2 
\end{array} \quad\longrightarrow\quad 
\frac{d}{dt} \vec{x} =
\begin{bmatrix}
	4 & -2 \\
	1 & 1
\end{bmatrix}
\vec{x}
$$

To find the solution to this system of ODEs, we first can start by applying the guess:

$$
\vec{x}(t) = \vec{v} e^{\lambda t}
$$

Putting this into the ODE system gives:

$$
\frac{d}{dt} \left(\vec{v} e^{\lambda t}\right) =
\begin{bmatrix}
	4 & -2 \\
	1 & 1
\end{bmatrix}
\left(\vec{v} e^{\lambda t}\right) \quad\implies\quad  \lambda \vec{v} e^{\lambda t} =
\begin{bmatrix}
	4 & -2 \\
	1 & 1
\end{bmatrix} \vec{v} e^{\lambda t}
$$

and rearranging after canceling out the exponentials:

$$
\begin{bmatrix}
	4 & -2 \\
	1 & 1
\end{bmatrix} \vec{v} = \lambda \vec{v} 
$$


We can Solve this eigenvalue problem by first finding the eigenvalues of the coefficient matrix using the characteristic equation:

$$
\begin{vmatrix}
	4 - \lambda & -2 \\
	1 & 1 - \lambda
\end{vmatrix} = (4 - \lambda)(1 - \lambda) - (-2) = 4 - 5 \lambda + \lambda^2 + 2 = \lambda^2 - 5 \lambda + 6 = 0
$$

This can be factored to get:

$$
(\lambda - 2)(\lambda - 3) = 0
$$

which gives the eigenvalues $\lambda_1 = 2$ and $\lambda_2 = 3$. 

From here we need to find the eigenvectors. This can be done by starting with the eigenvalue problem:

$$
\begin{bmatrix}
	4 - \lambda & -2 \\
	1 & 1 - \lambda
\end{bmatrix} \begin{bmatrix}
a \\ b 
\end{bmatrix} = \begin{bmatrix}
0 \\ 0
\end{bmatrix}
$$

where we have let $\vec{v} = \begin{bmatrix} a \\ b  \end{bmatrix}$. Taking the relation resulting from Row 2, we get:

$$ a + (1-\lambda)b = 0 \implies a = (\lambda - 1) b $$

Taking $a = 1$ gives:

$$ \vec{v}_i = \begin{bmatrix}
	\lambda_i - 1 \\ 1
\end{bmatrix} $$

So we have:

$$ \text{For: } \lambda_1 = 2 \qquad \vec{v}_1 = \begin{bmatrix}
1 \\1
\end{bmatrix} \qquad\qquad\qquad \text{For: } \lambda_2 = 3 \qquad \vec{v}_2 = \begin{bmatrix}
2 \\1
\end{bmatrix}$$

This leave us with the two solutions:

$$ \vec{x}_1(t) = \begin{bmatrix}
	1\\1
\end{bmatrix} e^{2 t} \qquad \text{and} \qquad  \vec{x}_2(t) = \begin{bmatrix}
2\\1
\end{bmatrix} e^{3 t} $$

Putting these together to create the general solutions leaves us with:

$$ \vec{x}(t) = C_1 \begin{bmatrix}
	1\\1
\end{bmatrix} e^{2 t} + C_2 \begin{bmatrix}
2\\1
\end{bmatrix} e^{3 t}   $$

In this example, both eigenvalues are positive, meaning the system grows over time in both eigen-directions.
{% endcapture %}
{% include example.html content=ex %}




{% capture ex %}
Consider the system of first-order ODEs:


$$
\begin{array}{l}
	\frac{d x_1}{dt} = 5 x_1 - 2 x_2 \\
	\frac{d x_2}{dt} = 3 x_1 - 4 x_2 
\end{array} \quad\longrightarrow\quad 
\frac{d}{dt} \vec{x} =
\begin{bmatrix}
	5 & -2 \\
	3 & -4
\end{bmatrix}
\vec{x}
$$


To find the solution to this system of ODEs, we first can start by applying the guess:

$$
\vec{x}(t) = \vec{v} e^{\lambda t}
$$

Putting this into the ODE system gives:

$$
\frac{d}{dt} \left(\vec{v} e^{\lambda t}\right) =
\begin{bmatrix}
	5 & -2 \\
	3 & -4
\end{bmatrix}
\left(\vec{v} e^{\lambda t}\right) \quad\implies\quad  \lambda \vec{v} e^{\lambda t} =
\begin{bmatrix}
	5 & -2 \\
	3 & -4
\end{bmatrix} \vec{v} e^{\lambda t}
$$

and rearranging after canceling out the exponentials:

$$
\begin{bmatrix}
	5 & -2 \\
	3 & -4
\end{bmatrix} \vec{v} = \lambda \vec{v} 
$$

We can Solve this eigenvalue problem by first finding the eigenvalues of the coefficient matrix using the characteristic equation:

$$
\begin{vmatrix}
	5 - \lambda & -2 \\
	3 & -4 - \lambda
\end{vmatrix} = (5 - \lambda)(-4 - \lambda) - (-6) = -20 - \lambda + \lambda^2 + 6 = \lambda^2 -  \lambda - 14 = 0
$$

This can be solved using the quadratic equation to get:

$$
\lambda = \frac{1 \pm \sqrt{(-1)^2 - 4(1)(-14)}}{2(1)} = \frac{1 \pm \sqrt{1 + 56}}{2} = \frac{1 \pm \sqrt{57}}{2}
$$

So, the two eigenvalues can be given as $ \lambda_1 = \frac{1 + \sqrt{57}}{2} $ and $\lambda_2 = \frac{1 - \sqrt{57}}{2} $.

From here we need to find the eigenvectors. This can be done by starting with the eigenvalue problem:

$$
\begin{bmatrix}
	5 - \lambda & -2 \\
	3 & -4 - \lambda
\end{bmatrix} \begin{bmatrix}
	a \\ b 
\end{bmatrix} = \begin{bmatrix}
	0 \\ 0
\end{bmatrix}
$$

where we have let $\vec{v} = \begin{bmatrix} a \\ b  \end{bmatrix}$. Taking the relation resulting from Row 1, we get:

$$ (5 - \lambda) a -2 b = 0 \implies b = \frac{1}{2} (5 - \lambda) a $$

Taking $a = 2$ gives:

$$ \vec{v}_i = \begin{bmatrix}
	2 \\ 5 - \lambda_i
\end{bmatrix} $$

So we have:

$$ \text{For: } \lambda_1 = \frac{1 + \sqrt{57}}{2} \qquad \vec{v}_1 = \begin{bmatrix}
	2 \\ \frac{9 - \sqrt{57}}{2}
\end{bmatrix} \qquad\qquad\qquad \text{For: } \lambda_2 = \frac{1 - \sqrt{57}}{2} \qquad \vec{v}_2 = \begin{bmatrix}
	2 \\ \frac{9 + \sqrt{57}}{2}
\end{bmatrix}$$

This leave us with the two solutions:

$$ \vec{x}_1(t) = \begin{bmatrix}
	2 \\ \frac{9 - \sqrt{57}}{2}
\end{bmatrix} e^{\left(\tfrac{1 + \sqrt{57}}{2}\right) t} \qquad \text{and} \qquad  \vec{x}_2(t) = \begin{bmatrix}
	2 \\ \frac{9 + \sqrt{57}}{2}
\end{bmatrix} e^{\left(\tfrac{1 - \sqrt{57}}{2}\right)  t} $$

Putting these together to create the general solutions leaves us with:

$$ \vec{x}(t) = C_1 \begin{bmatrix}
	2 \\ \frac{9 - \sqrt{57}}{2}
\end{bmatrix} e^{\left(\tfrac{1 + \sqrt{57}}{2}\right) t} + C_2 \begin{bmatrix}
2 \\ \frac{9 + \sqrt{57}}{2}
\end{bmatrix} e^{\left(\tfrac{1 - \sqrt{57}}{2}\right)  t}   $$


**Interpretation of the Solution**

Since $ \lambda_1 $ is positive and $ \lambda_2 $ is negative, this system exhibits **saddle point behavior**. That means:

- The mode corresponding to $ \lambda_1 $ grows exponentially. Meaning, if the initial condition aligns in anyway with $ \vec{v}_1 $, the solution will grow over time.
- The mode corresponding to $ \lambda_2 $ decays exponentially. Meaning, if the initial condition aligns with $ \vec{v}_2 $, the solution will decay toward zero.


In general, the system exhibits unstable behavior unless it starts purely in the decaying mode. In a real-world system—say, a mechanical or electrical system—this tells us that even a tiny perturbation along the unstable eigenvector will cause the solution to blow up exponentially over time. This is a common characteristic of unstable equilibrium points.
{% endcapture %}
{% include example.html content=ex %}

	






## Phase Portraits and Geometric Interpretation

Solving a system of ODEs analytically gives us explicit functions for $ x_1(t) $ and $ x_2(t) $, but the real insight often comes from **visualizing** how solutions behave over time. This is where **phase portraits** come into play. A phase portrait provides a geometric representation of all possible trajectories of a system in the state space.

A **phase portrait** is a plot of the solution trajectories in the $ (x_1, x_2) $-plane, also called the **phase plane**. Each curve in this plane represents a different solution of the system for a given initial condition. The direction of motion along these curves is indicated by arrows.

For a general system of linear ODEs:

$$
\frac{d}{dt} \vec{x} = \mathbf{A} \mathbf{x}
$$

where

$$
\vec{x} =
\begin{bmatrix}
	x_1 \\ x_2
\end{bmatrix}
\qquad\qquad
\mathbf{A} =
\begin{bmatrix}
	a & b \\
	c & d
\end{bmatrix}
$$

the phase portrait shows how the vector $ \vec{x} $ evolves in time.


As you can likely guess, the behavior of the system in it's phase portrait is largely dictated by the eigenvalues and eigenvectors of the matrix $ \mathbf{A} $:


- **Real, distinct eigenvalues:** The solution trajectories follow straight-line paths along the eigenvectors.
- **Complex eigenvalues:** The trajectories form spirals, indicating oscillatory motion.
- **Repeated eigenvalues:** The behavior depends on whether there is a full set of eigenvectors. If not, solutions follow a sheared structure.
- **Mixed signs of eigenvalues:** One mode grows while the other decays, producing a saddle point.


Phase portraits are useful because they allow us to understand the qualitative behavior of the system without solving explicitly for $ x_1(t) $ and $ x_2(t) $. They provide a way to predict stability, long-term behavior, and how solutions evolve for different initial conditions. In physical systems:


- A stable node represents a system that settles into equilibrium (e.g., a damped mass-spring system).
- A saddle point suggests an inherently unstable equilibrium (e.g., an inverted pendulum).
- A spiral suggests an oscillating system, such as electrical circuits or predator-prey models in biology.


To illustrate this, let's consider some common cases.


**Case 1: A Stable Node (Both Eigenvalues Negative)**

If the real portion of both eigenvalues of $ \mathbf{A} $ are negative, solutions will decay toward the origin as $ t \to \infty $.

Consider:

$$
\mathbf{A} =
\begin{bmatrix}
	-3 & -1 \\
	2 & -4
\end{bmatrix}
$$

Getting the eigenvalues:

$$
\begin{aligned}
	\begin{vmatrix}
	-3 - \lambda & -1 \\
	2 & -4 - \lambda
	\end{vmatrix} &= 0 \\
	(-3 - \lambda)(-4 - \lambda) - (-2) &= 0 \\
	\lambda^2 + 7 \lambda + 14   &= 0 
\end{aligned}
$$

which can be solved to get:

$$
\lambda_1 = \frac{1}{2} \big( -7 + i \sqrt{7} \big) \qquad \qquad \lambda_2 = \frac{1}{2} \big( -7 - i \sqrt{7} \big)
$$
	


<img
src="{{ '/courses/math-methods/images/lec19/Case 1.png' | relative_url }}"
alt="A grid-based plot with horizontal and vertical axes labeled x1 and x2. Red arrows across the plane point in different directions depending on location. In the upper region, arrows generally point downward, while in the lower region they point upward. The overall pattern shows motion that changes direction smoothly across the plane."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">







**Case 2: An Unstable Node (Both Eigenvalues Positive)**

If the real portion of both eigenvalues are positive, solutions tend away from the origin, meaning any small perturbation grows exponentially over time.


Consider:

$$
\mathbf{A} =
\begin{bmatrix}
	4 & 1 \\
	-2 & 3
\end{bmatrix}
$$

Getting the eigenvalues:

$$
\begin{aligned}
	\begin{vmatrix}
		4 - \lambda & 1 \\
		-2 & 3 - \lambda
	\end{vmatrix} &= 0 \\
	(4 - \lambda)(3 - \lambda) - (-2) &= 0 \\
	\lambda^2 - 7 \lambda + 14   &= 0 
\end{aligned}
$$

which can be solved to get:

$$
\lambda_1 = \frac{1}{2} \big( 7 + i \sqrt{7} \big) \qquad \qquad \lambda_2 = \frac{1}{2} \big( 7 - i \sqrt{7} \big)
$$
	
<img
src="{{ '/courses/math-methods/images/lec19/Case 2.png' | relative_url }}"
alt="A grid plot with axes labeled x1 and x2. Red arrows point outward from the center in many regions. In the right half, arrows tend to point to the right, while in the lower-right region they angle downward. The overall pattern suggests motion moving away from the center."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">




**Case 3: A Saddle Point (Eigenvalues of Opposite Signs)**

If the real portion of one eigenvalue is positive and the other is negative, the system exhibits **saddle behavior**, meaning some solutions approach the origin while others diverge.


Consider:

$$
\mathbf{A} =
\begin{bmatrix}
	2 & 3 \\
	3 & -2
\end{bmatrix}
$$

Getting the eigenvalues:

$$
\begin{aligned}
	\begin{vmatrix}
		2 - \lambda & 3 \\
		3 & -2 - \lambda
	\end{vmatrix} &= 0 \\
	(2 - \lambda)(-2 - \lambda) - (9) &= 0 \\
	\lambda^2 - 13   &= 0 
\end{aligned}
$$

which can be solved to get:

$$
\lambda_1 = +\sqrt{13} \qquad \qquad \lambda_2 = -\sqrt{13}
$$
	
<img
src="{{ '/courses/math-methods/images/lec19/Case 3.png' | relative_url }}"
alt="A grid plot with axes labeled x1 and x2. Red arrows form a curved pattern around the center. On the right side, arrows point upward and to the right, while on the left side they point downward and to the left. The pattern suggests a rotating or swirling motion."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">



**Case 4: Spiral Behavior (Complex Eigenvalues)**

If the eigenvalues are complex, solutions exhibit oscillatory motion. The real part determines whether the spirals decay (stable spiral) or grow (unstable spiral).

Consider:

$$
\mathbf{A} =
\begin{bmatrix}
	0 & -1 \\
	1 & 0
\end{bmatrix}
$$

Getting the eigenvalues:

$$
\begin{aligned}
	\begin{vmatrix}
		- \lambda & -1 \\
		1 & - \lambda
	\end{vmatrix} &= 0 \\
	\lambda^2 - (-1) &= 0 \\
	\lambda^2 + 1   &= 0 
\end{aligned}
$$

which can be solved to get:

$$
\lambda_1 = +i \qquad \qquad \lambda_2 = -i
$$
	
<img
src="{{ '/courses/math-methods/images/lec19/Case 4.png' | relative_url }}"
alt="A grid plot with axes labeled x1 and x2. Red arrows form a circular pattern around the center of the plot. The arrows are oriented so that motion follows a loop around the middle point, indicating consistent rotation."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">














## Inhomogeneous Systems and the Method of Undetermined Coefficients

So far, we’ve focused on **homogeneous** systems of ODEs of the form:

$$
\frac{d}{dt} \vec{x} = \mathbf{A} \vec{x}
$$

These systems describe the natural evolution of a system without any external inputs. But in many real-world applications, systems are **forced** by an external influence—be it an applied voltage in an electrical circuit, an external force in a mechanical system, or an input function in a control system.

To model these effects, we consider an **inhomogeneous** system:

$$
\frac{d}{dt} \vec{x} = \mathbf{A} \vec{x} + \vec{f}(t)
$$

where $ \vec{f}(t) $ represents the external forcing functions written as a vector. Our goal is to find a solution $ \vec{x}(t) $, which consists of two parts:

$$
\vec{x}(t) = \vec{x}_h(t) + \vec{x}_p(t),
$$

where, as we are familiar with already, $ \vec{x}_h(t) $ is the **homogeneous solution**, which solves $ \frac{d}{dt} \vec{x} = \mathbf{A} \vec{x} $, and $ \vec{x}_p(t) $ is the **particular solution**, which accounts for the inhomogeneous term $ \vec{f}(t) $.

### Choosing a Strategy: Why Use Undetermined Coefficients?

As we have seen throughout our discussion of solving ODEs, there are several methods for solving inhomogeneous systems. The most direct and practical f the approaches we discussed is the **method of undetermined coefficients**. Recall, this method works well when $ \vec{f}(t) $ has a simple structure, such as:


- Polynomials (e.g., $ \vec{f}(t) = \vec{a}t^2 + \vec{b}t + \vec{c} $)
- Exponentials (e.g., $ \vec{f}(t) = e^{\lambda t} \vec{a} $)
- Sines and cosines (e.g., $ \vec{f}(t) = \vec{a} \cos(\omega t) + \vec{b} \sin(\omega t) $)


The main idea behind the method is straightforward:

1) **Guess the form** of $ \vec{x}_p(t) $ based on $ \vec{f}(t) $.  
2) **Substitute** this guess into the differential equation.  
3) **Solve for the unknown coefficients**.


The reason this works is that if $ \vec{f}(t) $ has a simple form, then its derivatives will have a similar structure, meaning we can readily match terms and solve algebraically.


{% capture ex %}
Let’s apply this method to a concrete example. Consider the following system of ODEs:
	
$$
\begin{aligned}
\frac{dx_1}{dt} &= 2 x_1 + x_2 + e^t \\
\frac{dx_2}{dt} &= -3 x_1 + 4 x_2 + 2
\end{aligned} \quad\implies\quad \frac{d}{dt}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} =
\begin{bmatrix} 2 & 1 \\ -3 & 4 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} e^t \\ 2 \end{bmatrix}
$$

As per usual, we first find the homogeneous solution:

$$
\frac{d}{dt} \vec{x}_h = 
\begin{bmatrix} 2 & 1 \\ -3 & 4 \end{bmatrix} \vec{x}_h
$$

We find the eigenvalues of the coefficient matrix by solving:

$$
\begin{vmatrix} 2 - \lambda & 1 \\ -3 & 4 - \lambda \end{vmatrix}
=  (2 - \lambda)(4 - \lambda) - (-3) = \lambda^2 - 6\lambda + 11 = 0
$$

Solving for $ \lambda $ using the quadratic equation, we get the roots:

$$
\lambda = \frac{6 \pm \sqrt{36 - 44}}{2} = \frac{6 \pm i\sqrt{8}}{2} = 3 \pm i\sqrt{2}
$$

Since the eigenvalues are complex, the homogeneous solution consists of oscillatory terms:

$$
\vec{x}_h(t) = \begin{bmatrix} C_1\\ C_3 \end{bmatrix} e^{3t} \cos(\sqrt{2}t) + \begin{bmatrix} C_2 \\ C_4  \end{bmatrix} e^{3t} \sin(\sqrt{2}t) 
$$

Now we need to find the particular solution. Since the inhomogeneous term contains $ e^t $, we can guess an exponential function of the same form:

$$
\vec{x}_p(t) =
\begin{bmatrix} A e^t + B \\ C e^t + D \end{bmatrix}
$$

where we have to determine $A$, $B$, $C$, and $D$ using the differential equation. Taking the first derivative:

$$
\frac{d}{dt}
\begin{bmatrix} A e^t + B \\ C e^t + D \end{bmatrix} =
\begin{bmatrix} A e^t \\ C e^t \end{bmatrix}
$$

and substituting into the ODEs:

$$
\begin{bmatrix} A e^t \\ C e^t \end{bmatrix} =
\begin{bmatrix} 2 & 1 \\ -3 & 4 \end{bmatrix}
\begin{bmatrix} A e^t + B \\ C e^t + D \end{bmatrix} +
\begin{bmatrix} e^t \\ 2 \end{bmatrix}
$$

Expanding the matrix multiplication, adding the driving forces, and grouping like terms:

$$
\begin{bmatrix} A e^t \\ C e^t \end{bmatrix} =
\begin{bmatrix} 2A e^t + 2B + C e^t + D + e^t \\ -3A e^t - 3B + 4C e^t + 4D + 2 \end{bmatrix} =
\begin{bmatrix} \big(2A  + C + 1\big) e^t + \big(2B + D\big) \\ \big(-3A + 4C\big) e^t  + \big(- 3B  + 4D + 2\big) \end{bmatrix}
$$

We can solve for the undetermined coefficient by comparing like terms on either side of the equal sign.  Matching the $ e^t $ terms:

$$
A = 2A + C + 1 \qquad\qquad C = -3A + 4C
$$

and matching the constant terms:

$$
0 = 2B + D \qquad\qquad 0 = -3B + 4D + 2
$$

Simplifying al of the equations gives us the following system, in order of appearance:

$$
\begin{aligned}
	A + C &= -1 \\
	3A - 3 C &=0 \\
	2B + D &= 0 \\
	3B - 4 D &= 2 
\end{aligned}
$$

The second and third equations give:

$$
A = C \qquad \qquad D = -2 B
$$

Putting these into the first and fourth equations give us:

$$
A + (A) = -1 \implies A = - \frac{1}{2} \qquad \qquad 3 B -4(-2B) = 2 \implies B = \frac{2}{11}
$$

This leaves us with:

$$
A = - \frac{1}{2}, \quad C = - \frac{1}{2}, \quad B =  \frac{2}{11}, \quad D =  -\frac{4}{11}
$$

Thus, the particular solution is:

$$
\vec{x}_p(t) =
\begin{bmatrix} -\frac{1}{2}e^t + \frac{2}{11} \\ \frac{1}{2}e^t - \frac{4}{11} \end{bmatrix}
$$

The full solution is:

$$
\vec{x}(t) = \vec{x}_h(t) + \vec{x}_p(t) = \begin{bmatrix} C_1\\ C_3 \end{bmatrix} e^{3t} \cos(\sqrt{2}t) + \begin{bmatrix} C_2 \\ C_4  \end{bmatrix} e^{3t} \sin(\sqrt{2}t)  + \begin{bmatrix} -\frac{1}{2}e^t + \frac{2}{11} \\ \frac{1}{2}e^t - \frac{4}{11} \end{bmatrix}
$$
{% endcapture %}
{% include example.html content=ex %}

	
	









## Applications to Physics and Engineering

We now bring everything together by looking at how systems of ODEs naturally arise in physics and engineering. The framework of coupled differential equations shows up everywhere—from classical mechanics and electrical circuits to population dynamics and control systems. Understanding how to solve these systems is critical for modeling and predicting real-world behavior.

### Coupled Mass-Spring Systems

One of the simplest and most illustrative examples of a system of ODEs is a **coupled mass-spring system**. Suppose we have two masses, $ m_1 $ and $ m_2 $, connected by springs with constants $ k_1 $ and $ k_2 $, as shown below:


<img
src="{{ '/courses/math-methods/images/lec19/CoupledMasses.png' | relative_url }}"
alt="A horizontal mechanical system with a fixed wall on the left connected to a spring labeled k1. The spring is attached to a block labeled m1. A second spring labeled k2 connects that block to another block labeled m2 on the right. Arrows labeled x1 and x2 indicate the horizontal directions in which each block can move."
style="display:block; margin:1.5rem auto; max-width:600px; width:75%;">

Let $ x_1(t) $ and $ x_2(t) $ describe the displacements of the two masses from the left wall. By applying Newton’s Second Law to each mass, we obtain the system:

$$
\begin{array}{l}
	m_1 \frac{d^2 x_1}{dt^2} = -k_1 (x_1 - \ell_{0,1}) + k_2 (x_2 - x_1 - \ell_{0,2}) \\
	m_2 \frac{d^2 x_2}{dt^2} = -k_2 (x_2 - x_1 - \ell_{0,2})
\end{array}
$$

where $\ell_{0,1}$ and $\ell_{0,2}$ are the rest lengths of the springs, respectively. This equation can be simplified if we shift our coordinates to be that $ x_1(t) $ and $ x_2(t) $ describe the displacements of the two masses from their respective equilibrium positions. This translation of coordinates $x_1 \rightarrow x_1 + \ell_{0,1}$ and $x_2 \rightarrow x_2 + \ell_{0,1} + \ell_{0,2}$ gives us the following model:

$$
\begin{aligned}
	m_1 \frac{d^2 x_1}{dt^2} &= -k_1 (x_1 + \ell_{0,1} - \ell_{0,1}) + k_2 (x_2 + \ell_{0,1} + \ell_{0,2} - x_1 - \ell_{0,1} - \ell_{0 2}) \\
	m_2 \frac{d^2 x_2}{dt^2} &= -k_2 (x_2 + \ell_{0,1} + \ell_{0,2} - x_1 - \ell_{0,1} - \ell_{0,2})
\end{aligned} \quad\implies\quad \begin{aligned} m_1 \frac{d^2 x_1}{dt^2} &= -k_1 x_1 + k_2 (x_2 - x_1) \\
m_2 \frac{d^2 x_2}{dt^2} &= -k_2 (x_2 - x_1 )
\end{aligned}
$$

which is much easier to work with. 

Rewriting in matrix form:

$$
\begin{aligned}
	\frac{d^2 x_1}{dt^2} &= -\frac{k_1 + k_2}{m_1} x_1 + \frac{k_2}{m_1} x_2 \\
	\frac{d^2 x_2}{dt^2} &=  \frac{k_2}{m_2} x_1 - \frac{k_2}{m_2} x_2
\end{aligned} 
\quad\implies\quad 
\frac{d^2}{dt^2}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} =
\begin{bmatrix} -\frac{k_1 + k_2}{m_1} & \frac{k_2}{m_1} \\ \frac{k_2}{m_2} & -\frac{k_2}{m_2} \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

which can be simplified to

$$
\frac{d^2}{dt^2} \vec{x} =
\begin{bmatrix} -\frac{k_1 + k_2}{m_1} & \frac{k_2}{m_1} \\ \frac{k_2}{m_2} & -\frac{k_2}{m_2} \end{bmatrix} \vec{x}
$$

To solve this, we find the eigenvalues and eigenvectors of $ \mathbf{A} $, which tell us the natural frequencies and modes of oscillation. This system exhibits **normal modes**, where both masses oscillate in sync or out of phase.


{% capture ex %}
Let's work out a specific instance of this problem using the following relations:

$$
m_1 = m_2 = m \qquad \qquad k_1 = k_2 = k \qquad\qquad \frac{k}{m} = \omega_0^2
$$

This simplified the problem to:

$$
\frac{d^2}{dt^2} \vec{x} =
\begin{bmatrix} -2\omega_0^2 & \omega_0^2 \\ \omega_0^2 & -\omega_0^2 \end{bmatrix} \vec{x}
$$

We begin by guessing and exponential solution:

$$
\vec{x}(t) = \vec{v} e^{\omega t}
$$
where $\vec{v}$ is a constant vector. Putting this into the ODE gives:

$$
\omega^2 \vec{v} e^{\omega t} = \begin{bmatrix} -2\omega_0^2 & \omega_0^2 \\ \omega_0^2 & -\omega_0^2 \end{bmatrix} \vec{v} e^{\omega t} \quad\implies\qquad  \begin{bmatrix} -2\omega_0^2 & \omega_0^2 \\ \omega_0^2 & -\omega_0^2 \end{bmatrix} \vec{v} = \omega^2 \vec{v}
$$
which is an eigenvalue problem. Finding the eigenvalues:

$$
\begin{aligned}
	\begin{vmatrix} -2\omega_0^2 - \omega^2 & \omega_0^2 \\ \omega_0^2 & -\omega_0^2 - \omega^2 \end{vmatrix} &= 0   \\[1.25ex]
		(-2\omega_0^2 - \omega^2) (-\omega_0^2 - \omega^2) - \omega_0^4 &=0  \\[1.25ex]
	2\omega_0^4 +3 \omega_0^2 \omega^2 + \omega^4 - \omega_0^4  &=0 \\[1.25ex]
	\omega^4 + 3 \omega_0^2 \omega^2 + \omega_0^4  &=0
\end{aligned}
$$

Applying the quadratic equation gives:

$$
\omega^2 = \frac{-2\omega_0^2 \pm \sqrt{9\omega_0^4 - 4(1)(\omega_0^4)}}{2(1)} = \frac{-2\omega_0^2 \pm \sqrt{9\omega_0^4 - 4\omega_0^4}}{2} = \frac{-2\omega_0^2 \pm \omega_0^2 \sqrt{5}}{2}
$$

which gives us the following eigenvalues:

$$
\omega_1^2 = \Big(\frac{-2 +  \sqrt{5}}{2}\Big) \omega_0^2 \qquad\text{and}\qquad \omega_2^2 = \Big(\frac{-2 -  \sqrt{5}}{2}\Big) \omega_0^2
$$

Both of which are negative in value. Notice, this means $\omega_1$ and $\omega_2$ will both be imaginary numbers, meaning the solution to this problem will be oscillatory in nature.

Now we can find the eigenvectors. Beginning with the eigenvalue problem:

$$
\begin{bmatrix} -2\omega_0^2 & \omega_0^2 \\ \omega_0^2 & -\omega_0^2 \end{bmatrix} \vec{v} = \omega^2 \vec{v} \quad\implies\quad \begin{bmatrix} -2\omega_0^2 & \omega_0^2 \\ \omega_0^2 & -\omega_0^2 \end{bmatrix} \begin{bmatrix}
	a\\b
\end{bmatrix} = \omega^2 \begin{bmatrix}
	a\\b
\end{bmatrix}
$$

We can take the second row to get the following relation:

$$
\omega_0^2 a - \omega_0^2 b = \omega^2 b \quad\implies\quad a = \left(1 + \frac{\omega^2}{\omega_0^2}\right) b
$$

Letting $b = 2$ gives us the following eigenvectors:

$$
\text{For: }\omega_1^2 = \Big(\frac{-2 +  \sqrt{5}}{2}\Big) \omega_0^2 \qquad \vec{v}_1 = \begin{bmatrix}
	\sqrt{5} \\ 2
\end{bmatrix} 
\qquad\qquad
\text{For: }\omega_2^2 = \Big(\frac{-2 -  \sqrt{5}}{2}\Big) \omega_0^2 \qquad \vec{v}_2 = \begin{bmatrix}
	-\sqrt{5} \\ 2
\end{bmatrix}
$$

This gives us the following general solution:

$$
\vec{x}(t) = C_1 \vec{v}_1 \cos(\omega_1 t) + C_2 \vec{v}_1 \sin(\omega_1 t) + C_3 \vec{v}_2 \cos(\omega_2 t) + + C_4 \vec{v}_2 \sin(\omega_2 t)
$$

where $\omega_1$ and $\omega_2$ are the values found after removing out the imaginary $i$ when taking the square roots of their squared values found previously (i.e., the eigenvalues).
{% endcapture %}
{% include example.html content=ex %}








### Electrical Networks: The RLC Circuit

Another common example comes from electrical engineering. Consider a system where an inductor, resistor, and capacitor are connected in series with a voltage source:

$$
L \frac{dI}{dt} + RI + \frac{1}{C} \int I dt = V(t)
$$

Differentiating both sides:

$$
L \frac{d^2 I}{dt^2} + R \frac{dI}{dt} + \frac{1}{C} I = \frac{dV}{dt}
$$

This is a second-order linear ODE that resembles a damped harmonic oscillator. If we instead consider a circuit with multiple loops and coupled components, we end up with a system of ODEs.

For instance, consider two coupled RLC circuits:

$$
\begin{array}{l}
	L_1 \frac{dI_1}{dt} + R_1 I_1 + M \frac{dI_2}{dt} = V_1 \\
	L_2 \frac{dI_2}{dt} + R_2 I_2 + M \frac{dI_1}{dt} = V_2
\end{array}
$$

Here, $ M $ represents mutual inductance between the two circuits, which introduces coupling between their currents. Rewriting in matrix form:

$$
\begin{bmatrix} L_1 & M \\ M & L_2 \end{bmatrix}
\begin{bmatrix} \frac{dI_1}{dt} \\ \frac{dI_2}{dt} \end{bmatrix} +
\begin{bmatrix} R_1 & 0 \\ 0 & R_2 \end{bmatrix}
\begin{bmatrix} I_1 \\ I_2 \end{bmatrix} =
\begin{bmatrix} V_1 \\ V_2 \end{bmatrix}
$$

Again, solving this system requires eigenvalues and eigenvectors to understand how energy oscillates between the circuits.

### Population Models and Epidemiology

Not all applications involve mechanical or electrical systems. In biology and epidemiology, we often model interacting populations using systems of ODEs.

A classic example is the **Lotka-Volterra predator-prey model**, which describes interactions between two species:

$$
\begin{cases}
	\frac{dx}{dt} = \alpha x - \beta xy \\
	\frac{dy}{dt} = \delta xy - \gamma y
\end{cases}
$$

Here:

- $ x $ is the prey population (e.g., rabbits),
- $ y $ is the predator population (e.g., foxes),
- $ \alpha $ is the prey birth rate,
- $ \beta $ is the predation rate,
- $ \gamma $ is the predator death rate,
- $ \delta $ is the rate at which prey consumption leads to predator reproduction.


This system exhibits cyclic behavior, where predator and prey populations rise and fall periodically. The stability of these cycles depends on the eigenvalues of the Jacobian matrix evaluated at equilibrium points.

### Control Systems: Cruise Control in a Car

A more applied engineering example is automatic cruise control in a vehicle. The car's velocity $ v(t) $ is affected by the engine force $ u(t) $, air resistance, and road conditions. A simplified model is:

$$
m \frac{dv}{dt} = -b v + u(t)
$$

Introducing an error term $ e(t) = v_{\text{desired}} - v(t) $, we get a system where a control algorithm (such as a proportional-integral-derivative (PID) controller) adjusts $ u(t) $ to minimize $ e(t) $. This leads to a system of ODEs governing the closed-loop response.

















## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


Consider the system:

$$
\frac{d}{dt} \begin{bmatrix} x \\ y \end{bmatrix} =
\begin{bmatrix} -2 & 1 \\ 1 & -2 \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
$$


a) Find the eigenvalues and eigenvectors of the coefficient matrix.  

b) Classify the stability of the equilibrium point $ (0,0) $.  

c) Now suppose a driving force acts on the system giving you the equation:  

$$
\frac{d}{dt} \begin{bmatrix} x \\ y \end{bmatrix} =
\begin{bmatrix} -2 & 1 \\ 1 & -2 \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix} +
\begin{bmatrix} 0 \\ \cos(t) \end{bmatrix}
$$

Find the homogeneous solution using eigenvalues and eigenvectors.

d)  Use the method of undetermined coefficients to find a particular solution.  

e)  Write the general solution for $ \mathbf{x}(t) $.  

	








