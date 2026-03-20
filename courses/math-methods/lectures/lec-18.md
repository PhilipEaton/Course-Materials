---
layout: default
title: Mathematical Methods - Lecture 18
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 18
---


{% capture ex %}

{% endcapture %}
{% include example.html content=ex %}

# Lecture 18 – Laplace Transformations II



## Introduction: Continuing the Discussion

In the last lecture we were discussing the use of the Laplace Transformation in solving ODEs. Recall, the Laplace Transformation is defined in the following manner:

$$
\mathcal{L}\{f(t)\}(s) = F(s) = \int\limits_0^\infty e^{-st} f(t) \, dt
$$

and has one very important property for solving ODEs:

$$ \mathcal{L}\left\{\frac{d^n f}{dt^n}\right\}(s) = s^n \,  F(s) - s^{n-1} f(0)  - s^{n-2} \left.\frac{df}{dt}\right|_0 \dots - s \left.\frac{d^{n-2}f}{dt^{n-2}}\right|_0 - \left.\frac{d^{n-1}f}{dt^{n-1}}\right|_0   $$

it transforms derivatives in to an algebraic expression. This means we can take the Laplace transform of an ODE and turn it into an algebra problem. This leads itself to the following scheme for solving ODEs:


<img
src="{{ '/courses/math-methods/images/lec18/Chart.png' | relative_url }}"
alt="A flow chart showing how the laplace transformation gets around the issue of not being able to direction solve a differential equation by transforming it into an algebra problem, solving the algebra, and then taking the inverse of the transformation to get the solution."
style="display:block; margin:1.5rem auto; max-width:600px; width:70%;">

In the event, as indicated above, the ability to directly solve the ODE, taking the Laplace Transform can give us traction where otherwise we could end up stuck. 

Let's continue our discussion of this incredibly powerful tool by warming up with a simple physical application: a harmonic oscillator. 


{% capture ex %}
Consider a mass attached to a string without any friction:
	
$$ \frac{d^2x}{dt^2} = - \omega^2 x $$

We can take the Laplace transform of this equation to get:

$$ -x'(0) - s x(0) + s^2 \,  X(s) = - \omega^2 \, X(s) $$

We can take the following initial conditions $x(0) = x_0$ and $x'(0) = v_0$ -- the initial position and velocity, respectively. This gives:

$$ -v_0 - s x_0 + \left(s^2 + \omega^2\right) \,  X(s) = 0 \quad\implies\quad X(s)  = \frac{v_0 + s x_0}{s^2 + \omega^2} $$

We can go to a Laplace transform table to find the following transformations:

$$ \mathcal{L}\left[ \sin(a t) \right](s) = \frac{a}{s^2 + a^2} \hspace{1cm} \text{and } \hspace{1cm} \mathcal{L}\left[ \cos(a t) \right](s) = \frac{s}{s^2 + a^2}  $$

This allows us to take the inverse transform of both sides and write $x(t)$ in the following manner:

$$ x(t) =  \frac{v_0}{\omega} \,\, \mathcal{L}^{-1}\left\{\frac{\omega}{s^2 + \omega^2} \right\} + x_0 \,\, \mathcal{L}^{-1}\left\{\frac{s}{s^2 + \omega^2} \right\} \quad\implies\quad x(t) =  \frac{v_0}{\omega} \sin(\omega t) + x_0 \cos(\omega t)$$

We can extend this example by allowing there to be a dissipation term like linear air resistance or friction. This can be modeled using the following ODE:

$$ \frac{d^2x}{dt^2} = - \omega^2 x - 2\gamma \frac{dx}{dt}  $$

We can take the Laplace transform of this equation to get:

$$ -x'(0) - s x(0) + s^2 \,  X(s) = - \omega^2 \, X(s) - 2 \gamma \left( - x(0) + s \,  X(s)  \right) $$

We will again take the following initial conditions $x(0) = x_0$ and $x'(0) = v_0$ -- the initial position and velocity, respectively -- to get:

$$ -v_0 - s x_0 + s^2 X(s) =  - \omega^2  X(s) + 2 \gamma x_0 - 2 \gamma s \,  X(s)  \quad\implies\quad \Big(s^2 + 2 \gamma s + \omega^2\Big) X(s) =  v_0 + (s + 2 \gamma) x_0   $$

and solving for $X(s)$ gives:

$$  X(s) =  \frac{v_0}{s^2 + 2 \gamma s + \omega^2  } + \frac{(s + 2 \gamma) x_0}{s^2 + 2 \gamma s + \omega^2} $$

Going to the Laplace Transform table we can see some promising candidates for the inverse of these fractions:

$$ \mathcal{L}\left\{e^{-at} \sin(bt) \right\} =   \frac{b}{(s + a)^2 + b^2}  \quad ( s > -a ) $$ 

$$ \mathcal{L}\left\{e^{-at} \cos(bt) \right\} =   \frac{s + a}{(s + a)^2 + b^2}  \quad ( s > -a )$$

To make these fit we will have to complete the square for the denominator of our current solution: 

$$ s^2 + 2 \gamma s + \omega^2 \quad\implies\quad s^2 + 2 \gamma s + \gamma^2 - \gamma^2 + \omega^2 \quad\implies\quad (s + \gamma)^2 + \omega^2 - \gamma^2  $$

Applying this gives:

$$  X(s) =  \frac{v_0}{  (s + \gamma)^2 + \omega^2 - \gamma^2   } + \frac{(s + 2 \gamma) x_0}{  (s + \gamma)^2 + \omega^2 - \gamma^2  } $$

From this we can see that $ a = \gamma $ and $ b = \sqrt{\omega^2 - \gamma^2}$. WE have to do a minor rearrangement to get the inverse transformation to perfectly align:

$$  X(s) =  \frac{\sqrt{\omega^2 - \gamma^2}}{  (s + \gamma)^2 + \omega^2 - \gamma^2   } \,\,\, \frac{v_0 + \gamma x_0}{\sqrt{\omega^2 - \gamma^2}} + \frac{s + \gamma}{  (s + \gamma)^2 + \omega^2 - \gamma^2  } \,\,\, x_0 $$

Taking the inverse transform now gives:

$$ x(t) = e^{-\gamma t} \sin(\sqrt{\omega^2 - \gamma^2} \,\,\, t) \,\,\, \frac{v_0 + \gamma x_0}{\sqrt{\omega^2 - \gamma^2}} + e^{-\gamma t} \cos(\sqrt{\omega^2 - \gamma^2} \,\,\, t) \,\,\, x_0 $$

Finally, rearranging we have the solution:

$$ x(t) = \tfrac{v_0 + \gamma x_0}{\sqrt{\omega^2 - \gamma^2}}  \,\, e^{-\gamma t} \sin(\sqrt{\omega^2 - \gamma^2} \,\,\, t) + x_0 e^{-\gamma t} \cos(\sqrt{\omega^2 - \gamma^2} \,\,\, t) $$

Generally, to simplify this expression, we make the following substitution, $\omega_d = \sqrt{\omega^2 - \gamma^2}$ to get:

$$ x(t) = e^{-\gamma t} \left(\tfrac{v_0 + \gamma x_0}{\omega_d}  \, \sin(\omega_d t) + x_0 \cos(\omega_d t)\right) $$

{% endcapture %}
{% include example.html content=ex %}














## Step Functions and Discontinuous Inputs

So far, we’ve primarily dealt with differential equations where the forcing function (i.e., the inhomogeneous term in the ODE) is continuous and well-behaved. However, real-world systems often involve abrupt changes, such as turning a switch on or off, applying an instantaneous force (like kicking a ball), or engaging a control mechanism at a specific time (like rotors in self-righting robots). These situations introduce discontinuities, which we need a way to handle mathematically.

To model such behavior, we use the **Heaviside step function**, a tool that allows us to define piecewise functions in a clean and structured manner.

### The Heaviside Step Function

The **unit step function**, also called the Heaviside function and denoted $ U(t-c) $, is defined as:

$$
U(t-c) =
\begin{cases}
	0, & t < c, \\
	1, & t \geq c.
\end{cases}
$$

This function represents a switch that is off for all $t<c$ and turns on, and remains on at $ t = c $.

The Heaviside function allows us to describe piecewise-defined forcing functions compactly. For example, the function:

$$
f(t) =
\begin{cases}
	0, & t < 3, \\
	5, & t \geq 3
\end{cases}
$$

can be rewritten as:

$$
f(t) = 5 U(t-3)
$$

Or the more complicated function:

$$
g(t) =
\begin{cases}
	t, & t < 3, \\
	e^{-t}, & 3 \leq t \leq 8\\
	0, & t \geq 8
\end{cases}
$$

can be written as:

$$ g(t) = t \left(1 - U(t-3) \right) + e^{-t} \left(U(t-3) - U(t-8) \right) + 0 \, U(t - 8)  $$
$$ g(t) = t \left(1 - U(t-3) \right) + e^{-t} \left(U(t-3) - U(t-8) \right)  $$

where terms like $\left(U(t-3) - U(t-8) \right)$ turn on at $t-3$ and off at $t = 8$. 

This notation is incredibly useful when solving differential equations with Laplace transforms because step functions have a well-defined transform.

### Laplace Transform of the Step Function

A key reason step functions are useful is that their Laplace transform is straightforward:

$$
\begin{aligned}
	\mathcal{L} \{ U(t-c) \} &= \int_0^\infty e^{-st} U(t-c) \, dt  \\[1.5ex]
	&= \int_0^c e^{-st} U(t-c) \, dt + \int_c^\infty e^{-st} U(t-c) \, dt  \\[1.5ex]
	&= \int_0^c e^{-st} (0) \, dt + \int_c^\infty e^{-st} (1) \, dt  \\[1.5ex]
	&= \int_c^\infty e^{-st} \, dt  \\[1.5ex]
	&=  \left. -\frac{1}{s} e^{-st}\right|_c^\infty  \\[1.5ex]
	\mathcal{L} \{ U(t-c) \}  &=  \frac{1}{s} e^{-sc} \qquad (s > 0) 
\end{aligned}
$$

More generally, if we have a function $ g(t) $ that turns on at $ t = c $, we express it as $ g(t - c) U(t-c) $. Here $g(t-c)$ translates $g(t)$ to be centered on $t=c$ and the unit setp function it what turns it on at $t=c$ and keeps it off before that time. Its Laplace transform can be found to be:

$$
\begin{aligned}
	\mathcal{L} \{ g(t-c)U(t-c) \} &= \int_0^\infty e^{-st} g(t-c)U(t-c) \, dt  \\[1.5ex]
	&= \int_0^c e^{-st} g(t-c)U(t-c) \, dt + \int_c^\infty e^{-st} g(t-c)U(t-c) \, dt  \\[1.5ex]
	&= \int_0^c e^{-st} g(t-c)(0) \, dt + \int_c^\infty e^{-st} g(t-c)(1) \, dt  \\[1.5ex]
	&= \int_c^\infty e^{-st} g(t-c) \, dt  
\end{aligned}
$$

We can apply the $u$-sub of $u = t - c \implies du = dt$, to get:


$$
\begin{aligned}
	\mathcal{L} \{ g(t-c)U(t-c) \} &= \int_c^\infty e^{-st} g(t-c) \, dt  \\[1.5ex]
	&= \int_0^\infty e^{-s(u + c)} g(u) \, du  \\[1.5ex]
	&= e^{-sc} \int_0^\infty e^{-su} g(u) \, du  \\[1.5ex]
	\mathcal{L} \{ g(t-c)U(t-c) \}  &= e^{-sc} G(s)
\end{aligned}
$$

where $ G(s) $ is the Laplace transform of $ g(t) $. This result is essential for handling piecewise functions systematically.


{% capture ex %}
Let’s consider an example where a force is applied to a system at $ t = 3 $:

$$
y'' + 2y' + 5y = 10 U(t-3)
$$

with the initial conditions $y(0) = 0$ and $y'(0) = 0$. The could represent a damped simple harmonic oscillator that is started from rest at the equilibrium position and set into motion by a constant acceleration of $10$ m/s$^2$ acting to the right. 

**Step 1: Take the Laplace Transform**

Applying the transform to both sides:

$$
\mathcal{L} \{ y'' \} + 2\mathcal{L} \{ y' \} + 5\mathcal{L} \{ y \} = 10 \mathcal{L} \{ U(t - 3) \}
$$

Using known transforms:

$$
\Big(s^2 Y(s) - sy(0) - y'(0)\Big) + 2\Big(s Y(s) - y(0)\Big) + 5Y(s) = 10 \frac{e^{-3s}}{s}
$$

Applying the given initial conditions $ y(0) = 0 $ and $ y'(0) = 0 $, simplifies this to:

$$
s^2 Y(s) + 2 s Y(s)+ 5Y(s) = 10 \frac{e^{-3s}}{s} \quad\implies\quad (s^2 + 2s + 5) Y(s) = \frac{10 e^{-3s}}{s}
$$

**Step 2: Solve for $ Y(s) $**

Rearranging:

$$
Y(s) = \frac{10 e^{-3s}}{s(s^2 + 2s + 5)}
$$

**Step 3: Invert and get the solution $ y(t) $**


To invert this, we need to find the inverse of:

$$
\frac{10}{s(s^2 + 2s + 5)}
$$

since we have the following identify from the table of Laplace Transformations:

$$ \mathcal{L}^{-1}\Big\{ \frac{10 e^{-3s}}{s(s^2 + 2s + 5)} \Big\} = U(t-3)\mathcal{L}^{-1}\Big\{ \frac{10}{s(s^2 + 2s + 5)} \Big\} $$

The fraction we are interested in can be simplified using partial fraction decomposition and following that up with known inverse transforms. First, partial fractions:

$$
\frac{10}{s(s^2 + 2s + 5)} = \frac{A}{s} + \frac{Bs + C }{ s^2 + 2s + 5 } \quad\implies\quad 10 = A (s^2 + 2s + 5) + (Bs + C) s 
$$

Taking $s = 0$ gives $ 10 = A (0 + 0 + 5) + (B(0) + C) (0) \implies A = 2 $. So we have $10 = 2 (s^2 + 2s + 5) + (Bs + C) s $.

To find $B$ and $C$ we can pick two different values for $s$, like $s = 1$ giving $ 10 = 2 (1 + 2 + 5) + (B + C) \implies B + C = -6 $ and $s = -1$ giving $10 = 2 (1 - 2 + 5) - (-B + C) \implies -B + C =   -2 $. Give us the system of equations:

$$
\begin{aligned}
	B + C &= -6  \\
	-B + C &= -2  
\end{aligned}
$$

This can be solved to get $ C = -4$ and $B = -2 $. This leave us with the following decomposition:

$$ \frac{10}{s(s^2 + 2s + 5)} = \frac{2}{s} - \frac{2s + 4 }{ s^2 + 2s + 5 } $$

Completing the square in denominator of the second term gives us:

$$ \frac{10}{s(s^2 + 2s + 5)} = \frac{2}{s} - \frac{2s + 4 }{ (s+1)^2 + 4} = \frac{2}{s} - \frac{2(s + 1) }{ (s+1)^2 + 4} - \frac{2 }{ (s+1)^2 + 4}  $$

Taking the inverse of this gives:

$$ \mathcal{L}^{-1}\Big\{ \frac{10}{s(s^2 + 2s + 5)} \Big\} = 2 - 2 e^{-t}\cos(2t) - e^{-t}\sin(2t) $$

which leaves us with, remembering we have to shift the function to start at $t=3$:

$$
y(t) = U(t - 3) \Big( 2 - 2 e^{-(t-3)}\cos\big(2(t-3)\big) - e^{-(t-3)}\sin\big(2(t-3)\big) \Big)
$$

**Step 4: Interpret the Solution**

This result shows that before $ t = 3 $, there is no response ($ y = 0 $). After $ t = 3 $, the system oscillates with a damped sine/cosine wave, activated by the step function. Notice the steady state solution is $y_\text{Steady} = 2$, which means the system till settle slightly off center. This makes sense since the applied force is a constant force pushing to system in one direction. The systems settling in an off center position makes sense. 
{% endcapture %}
{% include example.html content=ex %}








## Application to Mechanical and Electrical Systems

Laplace transforms shine in physics and engineering because they provide a systematic way to solve differential equations that describe real-world systems. Mechanical oscillators, electrical circuits, and even control systems all follow the same mathematical principles—differential equations governing how a system responds to external inputs. Let's consider more examples of how physical systems respond to external driving forces of various types. 

### Mechanical Systems: The Damped Harmonic Oscillator

We have already considered a damped simple harmonic oscillator. We can extend this example by applying an external driving force which pushes the system away from its natural oscillation and into a new and interesting steady state. This type of system can be modeled using the following ODE

$$
m\frac{d^2x}{dt^2} + b\frac{dx}{dt} + kx = f(t)
$$

where:

- $ m $ is the mass,
- $ c $ is the damping coefficient,
- $ k $ is the spring constant, and
- $ f(t) $ is an external force acting on the system.


Typically we divide by the mass and redefine the terms in the following manner:

$$
\frac{d^2x}{dt^2} + \frac{b}{m} \frac{dx}{dt} + \frac{k}{m} \, x = \frac{1}{m} \, f(t) \quad\implies\quad  \frac{d^2x}{dt^2} + 2 \gamma \frac{dx}{dt} + \omega_0^2 \, x = \frac{1}{m} \, f(t)
$$

where:


- $ \gamma = \tfrac{b}{2m} $ is the dissipation constant, and
- $ \omega_0^2 = \tfrac{k}{m} $ is the natural frequency of oscillation for the undamped system.


For this example let's assuming zero initial conditions so that we are only considering how the system responds to the driving force $f(t)$. Taking the Laplace Transform gives:

$$
 s^2 X(s) + 2 \gamma s X(s) + \omega_0^2 X(s) = \frac{1}{m} F(s)
$$

which we can be solve for $X(s)$:

$$
X(s) = \frac{F(s)}{s^2 + 2\gamma s + \omega_0^2}
$$

It is generally simplest to complete the square in the denominator as that form is the one we will most commonly see in Laplace Transform tales. This gives:

$$
X(s) = \frac{F(s)}{(s + \gamma)^2 + \omega_0^2 - \gamma^2}
$$

This algebraic equation lets us directly analyze how the system responds to different types of forcing functions. For example, if $ F(t) $ is a step function (suddenly applying and sustaining a force), we can substitute $ F(s) = \frac{F_0}{s} $ and solve for $ X(s) $. The inverse Laplace transform then gives us the time-domain response.


{% capture ex %}
Let’s consider a unit impulse, $ F(t) = j \delta(t) $, where $j$ represents the amount of momentum transferred to the system (i.e., the impulse), applied to a mass-spring system without damping ($ \gamma = 0 $). Here $\delta(t)$ represents an instantaneous "kick" to the system at a time $t=0$. This function is called the Dirac-Delta Function. We will leave a detailed investigation into the properties of this function for a later lecture (Mathematical Physics) and must be satisfied for now with our simple conceptual understanding of this function--that being an instantaneous `kick'.  

The governing ODE for this situation can be given as:

$$
\frac{d^2x}{dt^2} + \omega_0^2 x = \frac{j}{m} \delta(t)
$$

Taking the Laplace transform:

$$
s^2 X(s) + \omega_0^2 X(s) = \frac{j}{m}
$$

Solving for $ X(s) $:

$$
X(s) = \frac{j/m}{s^2 + \omega_0^2}
$$

Comparing this to our Laplace Transform Table:

$$
X(s) = \frac{j/m}{\omega_0} \, \frac{\omega_0}{s^2 + \omega_0^2}
$$

we can see the inverse transform can be found to be:

$$
x(t) = \frac{j}{m\omega_0} \sin(\omega_0 t)
$$

This result tells us that an impulse applied at $ t = 0 $ causes the system to oscillate at its natural frequency. This is exactly the expected response: a sudden push on a mass-spring system results in free oscillations as if began from $x_0=0$ with some non-zero initial velocity $v_0 = \tfrac{j}{m}$. This makes sense since $j$ represented the momentum transferred via the `kick', which means $j/m$ is velocity the mass will possess as a result. 
{% endcapture %}
{% include example.html content=ex %}




### Electrical Circuits: RL, RC, and RLC Circuits

Laplace transforms also apply beautifully to electrical circuits, where Kirchhoff’s laws lead to differential equations describing voltage and current relationships. Consider the series RLC circuit:

$$
L\frac{d^2q}{dt^2} + R\frac{dq}{dt} + \frac{q}{C} = V(t).
$$

Here:

- $ L $ is inductance,
- $ R $ is resistance,
- $ C $ is capacitance, and
- $ V(t) $ is the input voltage.


Applying the Laplace transform, again assuming zero initial conditions so we are only considering the response of the system to the driving force:

$$
L s^2 Q(s) + R s Q(s) + \frac{1}{C} Q(s) = V(s)
$$

Rearranging:

$$
Q(s) = \frac{V(s)}{L s^2 + R s + \frac{1}{C}}
$$

This equation directly mirrors the mass-spring-damper system, showing a deep mathematical connection between mechanical oscillators and electrical circuits. By solving for $ Q(s) $, we can determine how charge, and thus current ($ I(s) = sQ(s) $), behaves in response to different driving voltages.

{% capture ex %}
Let’s examine a simple charging capacitor circuit where a resistor $ R $ and capacitor $ C $ are in series, and a step voltage $ V_0 U(t) $ is applied:

$$
R\frac{dq}{dt} + \frac{q}{C} = V_0 U(t)
$$

Taking the Laplace transform:

$$
R s Q(s) + \frac{1}{C} Q(s) = \frac{V_0}{s}
$$

Solving for $ Q(s) $:

$$
Q(s) = \frac{V_0}{s \left(R s + \frac{1}{C} \right)}
$$

Rewriting:

$$
Q(s) = \frac{V_0 / R}{s(s + 1/RC)} \quad\implies\quad Q(s) = \frac{I_0}{s(s + \omega_C^2)}
$$

where $I_0 = V_0/R$ is the initial current drawn when the voltage $V_0$ turns on, and $\omega_C^2 = \tfrac{1}{RC}$ is the natural frequency for the RC-circuit. 

Before we can take the inverse, we will need to apply partial fractions:

$$ \frac{I_0}{s(s + \omega_C^2)} = \frac{A}{s} + \frac{B}{s + \omega_C^2} \quad\implies\quad I_0 = A (s + \omega_C^2) + B s $$

Taking $s = 0$ gives $I_0 = A (0 + \omega_C^2) + 0 \implies A = \frac{I_0}{\omega_C^2} $ and taking $s = - \omega_C^2$ gives $I_0 = A (0) - B \omega_C^2 \implies B = - \frac{I_0}{\omega_C^2}  $ leaving us with:

$$ \frac{I_0}{s(s + \omega_C^2)} = \frac{I_0}{\omega_C^2} \left( \frac{1}{s} - \frac{1}{s + \omega_C^2} \right) $$

The coefficient in front can be simplified in the following manner $ \frac{I_0}{\omega_C^2} = \frac{I_0}{1/(RC)} = I_0 R C = V_0 C $. Using inverse Laplace transforms, this gives:

$$
q(t) = \frac{I_0}{\omega_C^2}  \left( 1 - e^{-\omega_C^2 t} \right) = C V_0 \left( 1 - e^{- t/RC} \right)
$$

Since $ I(t) = dq/dt $, the current response is:

$$
I(t) = \frac{V_0}{R} e^{-t/RC}
$$

This tells us that the capacitor charges up to $ CV_0 $ over time, with a characteristic time constant $ \tau = RC $, and that the initial charging current $ V_0/R $ decays exponentially.
{% endcapture %}
{% include example.html content=ex %}























## Problems:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.



**Problem 1: Solving an Electrical Circuit Problem**

Consider an RC circuit with a step input:

$$
R \frac{dq}{dt} + \frac{q}{C} = V_0 U(t).
$$

Given $ R = 2 $ ohms, $ C = 1 $ farad, and $ V_0 = 5 $ volts, solve for the charge $ q(t) $ using Laplace Transforms.

**Problem 2: Modeling an Impulse Response**

A mass-spring system satisfies:

$$
\frac{d^2x}{dt^2} + \omega_0^2 x = J \delta(t),
$$

where $ J $ is the impulse applied at $ t = 0 $. Solve for $ x(t) $ using Laplace Transforms and interpret your result.
















