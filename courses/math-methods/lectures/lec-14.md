---
layout: default
title: Mathematical Methods - Lecture 14
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 14
---


# Lecture 14 – Introduction to Second-Order ODEs


## Introduction

In many physical systems, dynamics are governed by equations that involve acceleration—that is, by second derivatives of position with respect to time--or, oddly enough, by the curvature if the fields involved--like potential energies. Unlike first-order ODEs, which typically capture rates of change, second-order ODEs can describe more complex behaviors such as oscillations and vibrations. This makes them indispensable in modeling systems ranging from mechanical oscillators (like mass-spring systems) to electrical circuits (such as RLC circuits), and even to more abstract phenomena in physics and engineering like the curvature of spacetime to explain gravity in General Relativity.

We will be begin by introducing second-order differential equations and reminding ourselves of a couple of important classifications. To start things off we will focus on the homogeneous, constant-coefficient,  second-order ODEs, where we will see a very general and effective solution strategy. Specifically, we will learn to solve homogeneous second-order ODEs using the **characteristic equation** method. As we explore the different cases—distinct real roots, repeated real roots, and complex conjugate roots—we will see how the form of the solution directly reflects the physical behavior of the system. For instance, the appearance of sine and cosine functions in the solution is directly related to oscillatory motion, while exponential terms indicate growth or decay.



## Classification of Second-Order ODEs

Second-order differential equations arise naturally in many physical systems, where the dynamics involve acceleration or other second-order effects. To effectively approach these equations, it is important to understand how they are classified. Let's remind ourselves of these categories and how to properly identify them.

### Homogeneous vs. Inhomogeneous Equations

**Homogeneous Equations:** 

A second-order ODE is said to be homogeneous if it can be written in the form

$$
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = 0
$$

**In a homogeneous equation, every term contains the function $ \boldsymbol{y} $ or one of its derivatives**. This type of equation typically describes the natural, unforced behavior of a system—like the free vibrations of a mass-spring system.

**Inhomogeneous Equations:** 

When an external force or input is present, the equation takes the form

$$
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = f(x),
$$

where $ f(x) $ is the inhomogeneous term representing a driving term. In this case, the solution is composed of a homogeneous part (the transient response) and a particular part (the steady-state response). Inhomogeneous equations are essential for modeling forced oscillations in mechanical systems or driven circuits in electrical engineering.


### Constant vs. Variable Coefficients

**Constant-Coefficient Equations:** 

These are the most common in introductory applications. The coefficients $ a $, $ b $, and $ c $ are constant:

$$
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = 0.
$$

The behavior of the physical system modeled using an equation like this can be oscillatory motion, exponential growth/decay, etc.

**Variable-Coefficient Equations:** 

In more complex situations, the coefficients depend on the independent variable $ x $: 

$$
a(x)\frac{d^2y}{dx^2} + b(x)\frac{dy}{dx} + c(x)\,y = 0.
$$

These equations are generally more difficult to solve and often require specialized techniques, such as power series methods. 


### Why do we care?
Knowing the  classification of an ODE helps determine the appropriate solution strategy. For example, homogeneous equations with constant coefficients are typically solved using the characteristic equation method, but this method is ineffective for homogeneous equations with variable coefficients. Further, understanding whether an equation is homogeneous or nonhomogeneous is crucial because it informs you about the structure of the solution—specifically, that the general solution will be the sum of a homogeneous solution and a particular solution.

These classifications lays the groundwork for solving second-order ODEs and connects directly to the physical interpretations of the solutions. Whether you’re modeling a free-falling mass, an oscillating spring, or an RLC circuit, knowing the structure of the equation helps you predict the behavior of the system before having to perform any complicated computations.












## Solving Homogeneous, Constant-Coefficient, Second-Order ODEs

In many physical systems, the behavior of a system is governed by a homogeneous second-order ODE with constant coefficients. Let's consider the general problem first:

$$
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = 0,
$$

where $a$, $b$, and $c$ are constants, and $x$ is the working variable. The key to solving these equations is to assume a solution of the form:

$$
y(x) = e^{rx}
$$

where $r$ is a **constant** to be determined. When you find a result for $r$ is **cannot** depend on the working variable. If it does, then you either made a mistake, or this strategy will not work for the ODE you are attempting to solve. 


**To be clear, this approach will work for any homogeneous, constant-coefficient, second-order ODE!**

>	This is one of the most popular and most useful approaches used when solving ODEs in physics and engineering!** When in doubt, try this approach. The worst that happens is you get the characteristic equation, notice the working variable is present, and realize this approach will not work. 


**Step 1: Find the Characteristic Equation**

We are going to want to make the substitution $y(x) = e^{rx}$ into the ODE. To do this we will need to know the first and second derivatives of this function:

$$
\frac{dy}{dx} = r e^{rx} \quad \text{and} \quad \frac{d^2y}{dx^2}  = r^2 e^{rx}
$$

Substituting these results into the ODE gives: 

$$
a r^2 e^{rx} + b r e^{rx} + c\,e^{rx} = 0
$$

Dividing the entire equation by $ e^{rx} $ (which is never zero) yields:

$$
a r^2 + b r + c = 0
$$

which we call the **characteristic equation**.

**Step 2: Solve the Characteristic Equation**

As with solving any quadratic equation, the roots can come in a variety of combinations. They can be real and distinct (no repeated roots), real and repeated, or they will be complex roots (specifically they will be complex conjugates of each other). We will have to look at each of these cases individually to see how we typically handle them:


- **Distinct Real Roots:** 
	If you solve for $r$ and find two unique results $ r_1 $ and $ r_2 $, then the general solution is:
	
  $$
	y(x) = C_1 e^{r_1 x} + C_2 e^{r_2 x}
	$$
	
  You create an exponential for each root, along with an unknown constant, and add them together.
	
- **Repeated Real Roots:** 
	If you solve for $r$ and find two repeated values $ r = r_1 = r_2 $, then the general solution is:
	
  $$
	y(x) = (C_1 + C_2 x)e^{rx}
	$$
	
  where one solutions is $C_1 e^{rx}$ and the other is $C_2 x e^{rx}$. The second solution actually comes from using variation of parameters, which we will investigate in a future lecture. For now it is fine to take this result at face value. 
	
- **Complex Conjugate Roots:** 
	In the event you solve for $r$ and find that is roots are complex conjugates of each other $r = \alpha \pm i \beta$, then we begin by doing the do the same thing as the real, distinct roots:
	
  $$
	y(x) = A_1 e^{(\alpha+i\beta)\,x} + A_2 e^{(\alpha-i\beta)\,x}
	$$
	
  Notice we can simplify this in the following manner:
	
  $$
	y(x) = A_1 e^{\alpha\,x} e^{i\,\beta x} + A_2 e^{\alpha\,x} e^{-i\,\beta x} \quad \implies \quad  y(x) = e^{\alpha\,x} \left( A_1  e^{i\,\beta x} + A_2 e^{-i\,\beta x}\right)
	$$
	
  Secondly, we can using the Euler Identity:
	
  $$
	e^{\pm i\theta} = \cos(\theta) \pm i \, \sin(\theta) 
	$$
	
  Using this in our general solution gives:
	
  $$
	y(x) = e^{\alpha\,x} \left( A_1  \left( \cos(\beta x) + i \, \sin(\beta x)  \right) + A_2   \left( \cos(\beta x) - i \, \sin(\beta x)  \right)   \right)
	$$
	
  and we can group on the trig functions:
	
  $$
	y(x) = e^{\alpha\,x} \left(    \left( A_1 + A_2\right)  \cos(\beta x) + i \left( A_1 - A_2\right) \, \sin(\beta x)  \right)
	$$
	
  Making the following change of unknown constants:
	
  $$ C_1 =  A_1 + A_2 \quad \text{and} \quad C_2 = i \left( A_1 - A_2\right)  $$
	
  gives us the following general solution:
	
  $$
	y(x) = e^{\alpha x}\left(C_1 \cos(\beta x) + C_2 \sin(\beta x)\right)
	$$


**Step 3: Physical Interpretation**

The form of the solution reveals much about the behavior of the system:

- For **distinct real roots**, the solution represents exponential growth or decay, which might model systems with over-damping.
- For **repeated real roots**, the presence of the $x$ term indicates a critical damping scenario, where the system returns to equilibrium without oscillating.
- For **complex conjugate roots**, the sine and cosine functions reveal oscillatory behavior, characteristic of underdamped systems. 


{% capture ex %}
**Real roots only produce exponential growth or decay.**
		
**Oscillatory behavior is indicated by complex roots to the characteristic equation.** 
{% endcapture %}
{% include result.html content=ex %}





{% capture ex %}
Consider the ODE for a **mass-spring system without damping**:

$$
\frac{d^2y}{dx^2} + \omega^2 y = 0
$$

This is a constant-coefficient, linear ODE. We can guess the following form for the solution:

$$ y(x) = e^{rx} $$

Taking the first and second derivatives:

$$ \frac{dy}{dx} = r e^{rx} \quad \text{and} \quad \frac{d^2y}{dx^2} = r^2 e^{rx}  $$

Putting this into the ODE gives:

$$
  r^2 e^{rx} + \omega^2 e^{rx} = 0
$$

Factoring out the common exponential

$$
e^{rx} \left( r^2  + \omega^2 \right)  = 0
$$

and demanding the parenthesis go to zero since the exponential can't, gives:

$$
r^2 + \omega^2 = 0
$$

This can be solved to give the roots $ r = \pm i\omega $. Notice these are complex, so the solution will contain sin and cos. Comparing this with the general solution we wrote down previously we can see $\alpha = 0$ and $\beta = \omega$. Hence, the general solution is:

$$
y(x) = C_1 \cos(\omega x) + C_2 \sin(\omega x)
$$

This solution perfectly captures the oscillatory nature of an undamped mass-spring system, where $ \omega $ is the angular frequency of oscillation.
{% endcapture %}
{% include example.html content=ex %}



### Initial Conditions

In all of the solutions we have obtained for second-order ODEs, we end up with two arbitrary constants. This should not surprise us since solving a second-order ODE requires integrating twice—each integration introduces a constant of integration. These constants reflect the generality of the solution; that is, without further information, there is a whole family of functions that satisfy the differential equation.

To pinpoint a unique solution that accurately describes a physical system, we need to specify two initial conditions. Typically, these are provided in the form:

$$
y(0) = y_0 \quad \text{and} \quad y'(0) = y'_0,
$$

where $ y_0 $ is the initial value of the function and $ y'_0 $ is the initial value of its derivative. In a physical context, these might correspond to the initial position and and initial velocity in a mechanical system, or to initial charge on and current through a capacitor in some electrical circuit.

Once these initial conditions are applied to the general solution, they allow us to solve for the two unknown constants, thereby yielding a particular solution that uniquely describes the system’s behavior. Understanding and applying these initial conditions is crucial—it not only ensures that the mathematical model is well-posed but also guarantees that the model reflects the real-world situation accurately.





## Physical Interpretation and Worked Examples

Let's look at some examples with actual numbers to help get a feel for how this process works, and then we will consider two general problems:


{% capture ex %}
Consider the following differential equation:

$$
\frac{d^2y}{dt^2} + 5\frac{dy}{dt} + 6y = 0
$$

This differential equation describes an overdamped oscillator--think a saloon door that could swing back and forth if it wasn't for the rust in the hinges that forces it to close slowly back into place. 

Assuming an exponential solution:

$$ y(x) = e^{rt} $$

gives us a characteristic equation of:

$$
r^2 + 5r + 6 = 0
$$

which factors as:

$$
(r + 2)(r + 3) = 0
$$

The roots are real and distinct:

$$
r = -2 \quad \text{and} \quad r = -3
$$

The general solution is:

$$
y(t) = C_1 e^{-2t} + C_2 e^{-3t}
$$

This solution represents an overdamped system, where the response decays exponentially to zero without oscillation. In physical terms, the system slowly returns to equilibrium, with the rate of decay determined by the two negative roots.
{% endcapture %}
{% include example.html content=ex %}



{% capture ex %}
A critically damped system is one where the system returns to equilibrium as quickly as possible without oscillating. For instance it could be useful for a door leading outside to be critically damped. That way when people enter and leave the door closes as fast a possible, keeping the warm air in during winter and the hot air out during summer.  Another great example is shock absorbers. 

Consider the following critically damped system:

$$
\frac{d^2y}{dt^2} + 4\frac{dy}{dt} + 4y = 0
$$

Assuming an exponential solution:

$$ y(x) = e^{rt} $$

gives us a characteristic equation of:

$$
r^2 + 4r + 4 = 0 \quad \implies \quad (r + 2)^2 = 0
$$

Here, we have a repeated real root:

$$
r = -2
$$

The general solution then takes the form:

$$
y(t) = \left(C_1 + C_2 t\right) e^{-2t}
$$
{% endcapture %}
{% include example.html content=ex %}







**Couple of comments about these examples:**

In these examples, the constants $C_1$ and $C_2$ are determined by the initial conditions of the system (e.g., initial position and velocity). These constants tailor the general solution to the specific scenario at hand, dictating the system’s **transient response** before it settles into its **long-term, steady-state response**.







## Transient and Steady-State (Particular) Responses

When solving differential equations, especially in systems subject to external driving forces, it's important to understand that the overall response of the system is typically composed of two distinct parts: the **transient response** and the **steady-state response** (or forced response or particular response).


### Transient Response

The transient response is **the short-term response of the system to its initial conditions**. It is captured by **the solution to the homogeneous equation** (i.e., the part of the solution that arises from the initial conditions). In many physical systems, transient effects decay over time due to damping or other dissipative processes. This component describes how the system reacts immediately after a disturbance, but its influence fades as the system settles into its long-term behavior.

### Steady-State (Particular) Response

The steady-state response, on the other hand, is the long-term behavior of the system that persists once the transient effects have died out. This response is driven solely by the external forcing function in the differential equation. In practical terms, the steady-state solution tells us how the system behaves under continuous, persistent input—like the constant charge on a capacitor in an RC circuit, or the sustained oscillation in a forced mechanical system. This response is captured by **the particular solution to an inhomogenous differential equation**, which we will consider in more detail in the next lecture.

{% capture ex %}
Consider a forced oscillatory system described by:

$$
\frac{d^2y}{dt^2} + 2 \gamma  \frac{dy}{dt} + \omega_0^2 y = F(t)
$$

where $\gamma$ is the damping constant, $\omega_0$ is the natural frequency, and $F(t)$ is the external driving force. The general solution to this inhomogeneous problem, as we saw with first-order linear ODEs, can be split into a homogeneous solution $y_h(t)$ and the particular solution $y_p(t)$ for the driving term $F(t)$ giving us:

$$
y(t) = y_h(t) + y_p(t)
$$

where $y_h(t)$ is the  homogeneous (transient) solution and $y_p(t)$ is the particular solution. 

As $ t \to \infty $, the transient component $y_h(t)$ decays to zero (provided the system is damped), leaving only the steady-state solution $y_p(t)$ that fully describes the long-term behavior of the system.

If the system is undamped, additional procedures—such as averaging over one natural period—can be applied to effectively remove the transient component of the solution. This process isolates the steady-state response, which represents only the effects of the external driving force.

A classic example of this approach is found in celestial mechanics. While planets continuously orbit around the system's center of mass, there are situations where we are interested in how an external force changes that motion. For instance, when studying the orbit of Mercury around the Sun, you might want to determine the impact of Jupiter's gravitational pull on Mercury's trajectory. In this case, Jupiter's gravitational force acts as an external driving force on the Sun–Mercury system, and averaging over the natural orbital period allows us to filter out the inherent undamped motion, thereby highlighting the perturbative effects.
{% endcapture %}
{% include example.html content=ex %}








## Energy Considerations and Phase Space Analysis}

Another way to connect the mathematics of second-order ODEs to the physical behavior of a system is by considering energy and phase space. In many mechanical systems, such as mass-spring oscillators, energy conservation (or dissipation in the presence of damping) plays a crucial role in determining the system’s dynamics.


### Energy Considerations

For an undamped mass-spring system, the total mechanical energy—composed of kinetic energy $ \frac{1}{2} m \left(\frac{dx}{dt}\right)^2 $ and potential energy $ \frac{1}{2} k x^2 $—remains constant. We could use this to develop an ODE by writing out the total mechanical energy:

$$ E = KE + PE \quad \implies \quad E = \frac{1}{2} m \left(\frac{dx}{dt}\right)^2 +  \frac{1}{2} k x^2 $$

and then take the time derivative of the whole thing. Since the energy is conserved if there is no damping we get:

$$ \frac{dE}{dt} = \frac{d}{dt} \left( \frac{1}{2} m \left(\frac{dx}{dt}\right)^2 +  \frac{1}{2} k x^2\right)  \quad \implies \quad 0 =  m \left(\frac{dx}{dt}\right) \left(\frac{d^2x}{dt^2}\right) +  k x\left(\frac{dx}{dt}\right) $$

which we can divide by the velocity to get:

$$  m \frac{d^2x}{dt^2} +  k x = 0 $$

When damping is introduced, energy is gradually lost due to an external work, like friction. This is reflected in the exponential decay of the homogeneous solution. This decay represents the dissipation of energy over time, explaining why the transient response fades away. We can incorporate this into our energy problem by recalling that work is written in the following manner:

$$ W = \int \vec{F} \cdot d\vec{x} $$

which can be changed to an integration on time via a chain rule:

$$ W = \int \vec{F} \cdot \frac{d\vec{x}}{dt} \, dt $$ 

In one dimension this would be:

$$ W = \int F \frac{dx}{dt} \, dt $$ 

If we assume the force if a drag force of the same form we used previously, we would be left with:

$$ W = \int -b \left(\frac{dx}{dt}\right)^2 \, dt $$ 

In this case any changes in the energy would be a result of this work. Taking the time derivative gives:

$$ \frac{dW}{dt} =  -b \left(\frac{dx}{dt}\right)^2 $$

how the total energy will change in time. Using this in the above derivative of the total mechanical energy gives:

$$ -b \left(\frac{dx}{dt}\right)^2 =  m \left(\frac{dx}{dt}\right) \left(\frac{d^2x}{dt^2}\right) +  k x\left(\frac{dx}{dt}\right) $$

Canceling out a velcity and rearranging gives:

$$ m \frac{d^2x}{dt^2}  + b \frac{dx}{dt} +  k x = 0 $$

the ODE we have been working with for a damped simple harmonic oscillator. 



### Phase Space Analysis

Another valuable tool is the phase space diagram, where we plot the displacement $ x $ against the momentum $ m\frac{dx}{dt} $. This graphical representation offers a direct view of the system’s evolution:

- In an undamped system, the phase trajectory forms a closed curve (typically an ellipse) representing perpetual oscillations.
	
<img
src="{{ '/courses/math-methods/images/lec14/Phase Space Undamped.png' | relative_url }}"
alt="Horizontal lines at y = 0 and y = 1."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">
	
- In a damped system, the trajectory spirals inwards towards the equilibrium point, reflecting the gradual loss of energy.
	
<img
src="{{ '/courses/math-methods/images/lec14/Phase Space Damped.png' | relative_url }}"
alt="Horizontal lines at y = 0 and y = 1."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">



Phase space diagrams not only provide insight into the transient and long-term behavior but also help to visualize how different damping regimes (underdamped, critically damped, overdamped) affect the system dynamics.







## General Damped Oscillator Problem

Let's consider the general problem of a damped simple harmonic oscillator. This can be nicely model by assuming there is a spring force ($F_s = - k x$) acting on an object of mass $m$, and a drag force ($F_d = - b v$), where $x$ and $v$ are the displacement from equilibrium and velocity, respectively. Putting these into Newton's second law:

$$
\begin{aligned}
	F_{\text{net}} &= F_s + F_d \[0.75ex]
	m \, a &= - k x - b v\[0.75ex]
	m \, \frac{d^2x}{dt^2} &= - k x - b \frac{dx}{dt}
\end{aligned}
$$

Putting this into standard form gives:

$$ m \, \frac{d^2x}{dt^2} + b \frac{dx}{dt} + k x  = 0 $$

a homogeneous, constant-coefficient, linear, second-order ODE. We can solve this by assuming an exponential solution:

$$ x(t) = e^{rt} $$

which gives a characteristic equation of:

$$ m r^2 + b r + k = 0 $$

Applying the quadratic equation gives:

$$ r = \frac{-b \pm \sqrt{b^2 - 4(m)(k)}}{2 m} $$

which can be simplified to:

$$ r = -\frac{b}{2m} \pm \sqrt{\left(\frac{b}{2m}\right)^2 - \frac{k}{m}} $$

To simplify things, let's call $\gamma = \frac{b}{2m}$ and $\omega_0^2  = \frac{k}{m}$, the damping constant and natural angular frequency, respectively. This allows us to write the original ODE as:

$$ \frac{d^2x}{dt^2} + 2\gamma \frac{dx}{dt} + \omega_0^2 x  = 0 $$

and the roots of the characteristic equation as:

$$ r = -\gamma \pm \sqrt{\gamma^2 - \omega_0^2} $$

There are three possible outcomes:

- **Real, Discrete Roots**: This implies the square root gives a real, non-zero result. This is only possible if $$\gamma^2 > \omega_0^2 \implies \gamma > \omega_0$$ This is referred to as **overdamped** since the damping constant is larger than the natural frequency. 
- **Real, Repeated Roots**: This implies the square root is zero. This is only possible if $$\gamma^2 = \omega_0^2 \implies \gamma = \omega_0$$ This is referred to as **critically damped** since the damping constant is exactly equal to the natural frequency.
- **Complex Roots**: This implies the square root gives an imaginary, non-zero result. This is only possible if $$\gamma^2 < \omega_0^2 \implies \gamma < \omega_0$$ This is referred to as **underdamped** since the damping constant is smaller than the natural frequency. 

<img
src="{{ '/courses/math-methods/images/lec14/Damped Oscillator.png' | relative_url }}"
alt="Horizontal lines at y = 0 and y = 1."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">






### Overdamped Oscillator

This is the case where $\gamma > \omega_0$, where the roots of the characteristic equation are real and not repeated, the general solution will be two decaying exponential solutions. Specifically:

$$ x(t) = A_1 e^{(-\gamma + \sqrt{\gamma^2 - \omega_0^2} ) \, t } + A_2 e^{(-\gamma - \sqrt{\gamma^2 - \omega_0^2} ) \, t} $$

where $A_1$ and $A_2$ are unknown constants that will need to be determined via initial conditions. 

Picking some initial condition and plotting the phase space diagram of this solutions give a plot that looks something like:


<img
src="{{ '/courses/math-methods/images/lec14/Phase Overdamped.png' | relative_url }}"
alt="Horizontal lines at y = 0 and y = 1."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

Notice this does not oscillate. It just goes to rest immediately. 





### Critically Damped Oscillator

This is the case where $\gamma = \omega_0$, where the roots of the characteristic equation are real and repeated, the general solution will be a decaying exponential solution with a linear polynominal as its coefficient. Specifically:

$$ x(t) = (B_1 + t \, B_2) e^{-\gamma t} $$

where $B_1$ and $B_2$ are unknown constants that will need to be determined via initial conditions. 

Picking some initial condition and plotting the phase space diagram of this solutions give a plot that looks something like:

<img
src="{{ '/courses/math-methods/images/lec14/Phase Critical.png' | relative_url }}"
alt="Horizontal lines at y = 0 and y = 1."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

This looks very similar to the over damped plot, but it does takes a slightly longer path to get to rest. This is the optical phase path to get to rest in the least amount of time. 



### Underdamped Oscillator

This is the case where $\gamma < \omega_0$, where the roots of the characteristic equation are complex and are conjugates of each other. The general solution, as we have seen, will be sine and cosine with a decaying exponential. Specifically:

$$ x(t) = \left(C_1 \, \cos\left( \sqrt{\omega_0^2 - \gamma^2}  \, \, t\right)  + C_2 \sin\left( \sqrt{\omega_0^2 - \gamma^2}  \, \, t \right) \right) e^{-\gamma t} $$

where $C_1$ and $C_2$ are unknown constants that will need to be determined via initial conditions. 

Picking some initial condition and plotting the phase space diagram of this solutions give a plot that looks something like:

<img
src="{{ '/courses/math-methods/images/lec14/Phase Underdamped.png' | relative_url }}"
alt="Horizontal lines at y = 0 and y = 1."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

Notice this circles around and around decaying as it goes until it comes to rest at $x = 0$ and $v = 0$. 



## General RLC Oscillator Problem

Let's consider the general problem of a circuit with a resistor (resistance $R$), an inductor (inductance $L$), and a capacitor (capacitance $C$) are all hooked up in series. Applying a Kirchhoff loop rule to this circuit will give the following voltage equation

$$ L \, \frac{dI}{dt} + R \, I + \frac{1}{C} \, Q(t)  = 0 $$

where $I(t)$ is the current in the circuit and $Q(t)$ is the charge stored on the capacitor. Through a bit of logic we can convince ourselves that the current into the capacitor will be related to the rate at which the charge of the capacitor changes:

$$ I(t) = \frac{dQ}{dt} $$

which allows us to write the ODE as:

$$ L \, \frac{d^2 Q}{dt^2} + R \, \frac{dQ}{dt} + \frac{1}{C} \, Q  = 0 $$

This is a homogeneous, constant-coefficient, linear, second-order ODE. We can solve this by assuming an exponential solution:

$$ Q(t) = e^{rt} $$

which gives a characteristic equation of:

$$ L r^2 + R r + \frac{1}{C} = 0 $$

Applying the quadratic equation gives:

$$ r = \frac{-R \pm \sqrt{R^2 - 4(L)(\tfrac{1}{C})}}{2 L} $$

which can be simplified to:

$$ r = -\frac{R}{2L} \pm \sqrt{\left(\frac{R}{2L}\right)^2 - \tfrac{1}{CL}} $$

To simplify things, let's call $\gamma = \frac{R}{2L}$ and $\omega_0^2  = \frac{1}{CL}$, the damping constant and natural angular frequency, respectively. This allows us to write the original ODE as:

$$ \frac{d^2Q}{dt^2} + 2\gamma \frac{dQ}{dt} + \omega_0^2 Q  = 0 $$

and the roots of the characteristic equation as:

$$ r = -\gamma \pm \sqrt{\gamma^2 - \omega_0^2} $$

Notice, this is identical in form to the damped simple harmonic oscillator! That means the same three siutaiton are possible -- overdamped, critically damped, and underdamped -- and the form of the general solutions are the same! 







## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.



Consider a mass-spring-damper system governed by the differential equation:
$$
m \frac{d^2x}{dt^2} + b \frac{dx}{dt} + kx = 0,
$$
where $ m $ is the mass, $ b $ is the damping coefficient, and $ k $ is the spring constant. For this problem, assume:
$$
m = 1 \, \text{kg}, \quad b = 2 \, \text{N}\cdot\text{s/m}, \quad k = 5 \, \text{N/m}.
$$

**Tasks:**

a) Write the differential equation for this system using the given parameters.  

b) Derive the characteristic equation associated with this ODE and solve for its roots.  

c) Based on the roots, classify the system as underdamped, critically damped, or overdamped. Explain your reasoning.  

d) Find the general solution $ x(t) $ for the system.  

e) Discuss the physical interpretation of your solution. In particular:  

- How do the damping and stiffness affect the behavior of the system?
- What is the significance of the transient response in this context?
- How would the long-term behavior (steady state) of the system appear?











