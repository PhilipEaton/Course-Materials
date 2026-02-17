---
layout: default
title: Mathematical Methods - Lecture 11
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 11
---



# Lecture 11 – Introduction of Ordinary Differential Equations


## Why Do We Need Differential Equations?

In physics and engineering, we often encounter systems that evolve in time, space, or both simultaneously. Whether we are tracking the motion of a falling object, the oscillations of a spring, or the rate at which a chemical reaction progresses, these processes are not best described by simple algebraic equations. Instead, they are governed by **differential equations**—**equations that relate a function to its own rate of change**.

At their core, differential equations describe how something *changes*. If we know how a system evolves at each instant, we can predict its future behavior by solving an appropriate differential equation. These equations form the backbone of physics, and virtually every other field of applied science. 

To motivate our study, let’s consider some key examples where differential equations naturally arise in physics.

**Newton’s Second Law: Classical Mechanics**

The fundamental equation of motion in classical mechanics states that the acceleration of an object is proportional to the net force acting on it:

$$
m \frac{d^2\vec{r}}{dt^2} = \vec{F}_\text{net}(\vec{r}, \vec{v}, t).
$$

This is a **second-order differential equation** because it involves the second derivative of position $ \vec{r}(t) $ with respect to time. Later in this chapter, we will formalize how we classify different types of differential equations.

A simple example is a mass attached to a spring, where the restoring force is given by Hooke’s Law:

<img
  src="{{ '/courses/math-methods/images/lec11/Lecture11Fig1.png' | relative_url }}"
  alt="Free body diagram of a mass on a spring. Gravitational force points down, normal force up, and spring force to the left."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">


For this situation we can assume, for simplicity, that the normal force and gravitational force cancel out:  

$$
N = W = mg
$$

According to the forces drawn in the figure to the left, the net force in the horizontal direction is entirely due to the restoring force of the spring:

$$
F_{\text{net}, x} = F_{s, x} = - k x
$$

where $ x $ is the displacement from equilibrium. Applying Newton’s Second Law gives:

$$
m \frac{d^2x}{dt^2} = - k x.
$$


This equation tells us that the acceleration of the mass depends on its displacement from equilibrium. Importantly, we see a direct link between the displacement of the mass and its second derivative with respect to time—meaning this is a differential equation.


**Electrical Circuits: The RLC Circuit**

In an electrical circuit containing a resistor ($ R $), inductor ($ L $), and capacitor ($ C $), Kirchhoff’s voltage law leads to the following differential equation for the charge stored on the capacitor $ Q(t) $:

$$
L \frac{d^2Q}{dt^2} + R\frac{dQ}{dt}  + \frac{1}{C} Q = V(t)
$$

This equation describes how the charge on the capacitor evolves in response to an applied voltage $ V(t) $. 



**Radioactive Decay: Exponential Decay Law**

In nuclear physics, the number of unstable nuclei in a sample decreases at a rate proportional to the number of remaining nuclei:

$$
\frac{dN}{dt} = -\lambda N
$$

This is a **first-order differential equation**, where $ \lambda $ is the decay constant. We will show that the solution to this equation is an exponential function, describing how radioactive substances decay over time. This type of equation also appears in chemical reactions and population dynamics.



**Fluid Flow and Heat Transfer: Partial Differential Equations**

More complex physical systems involve **partial differential equations (PDEs)**, which contain derivatives with respect to multiple variables. For example, the **heat equation** describes the transfer of thermal energy in a material:

$$
\frac{\partial T}{\partial t} = \alpha \frac{\partial^2T}{\partial x^2}
$$

This equation describes how temperature $ T(x,t) $ evolves in space and time due to thermal diffusion.

While we focus on **ordinary differential equations (ODEs)** in this chapter, PDEs play a major role in many areas of physics and engineering.



**Quantum Mechanics: The Schrödinger Equation**

Perhaps one of the most profound differential equations in physics is the Schrödinger equation, which governs quantum mechanical systems:

$$
i \hbar \frac{d}{dt} |\psi\rangle = \hat{H} |\psi\rangle
$$

Here, $ |\psi\rangle $ is the quantum state of the system, and $ \hat{H} $ is the Hamiltonian operator, which is related to the energy of the system. The eigenvalues of $ \hat{H} $ correspond to the possible energy levels of the system.

This equation tells us how a quantum state evolves over time and is central to our understanding of atomic, molecular, and condensed matter physics.




## Classification of Ordinary Differential Equations

Before we begin solving differential equations, we first need to understand how they are classified. Not all differential equations are the same—some are simple to solve, while others require more sophisticated techniques. Properly classifying differential equations helps tell us something about how difficult it will be to solve, which techniques we should consider applying, and what kinds of physical systems it can be used to describe.

### Ordinary vs. Partial Differential Equations

**Ordinary differential equations (ODEs)** involve derivatives with respect to a single working variable (e.g., time), which can be useful for describing the motion of an object, for example. However, many physical systems require an understanding of how something changes over time *and* space. Differential equations that involve derivatives of multiple variables are called **partial differential equations (PDEs)**. 

As we saw in the previous section, the one-dimensional heat equation:

$$
\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}
$$

describes how temperature $ T(x, t) $ evolves over time in a one-dimensional space. 

PDEs are a topic for another time, but it is useful to recognize their role in physics. We will focus our attention on ODEs because they form the foundation for solving PDEs later on.



### Order of a Differential Equation

The **order** of a differential equation is determined by the **highest derivative** that appears in the equation. For example:

- A **first-order** differential equation involves only the first derivative:
	
	$$
	\frac{dx}{dt} = f(x, t)
	$$
	
- A **second-order** differential equation involves the second derivative:
	
	$$
	m \frac{d^2x}{dt^2} = -k x
	$$


The order of a differential equation provides insight into how many times we will need to integrate in order to obtain a solution. A first-order ODE requires one integration, while a second-order ODE requires two integrations to fully remove the derivatives.

For example, we can consider the simple differential equation:

$$
\frac{dx}{dt} = b
$$

where $ b $ is a constant. This equation states that the rate of change of $ x $ is a constant value $ b $. To solve for $ x(t) $, we can integrate both sides with respect to $ t $:

$$
\int \frac{dx}{dt} \, dt = \int b \, dt
$$

By the fundamental theorem of calculus, the integration cancels the derivative, leaving us with:

$$
x(t) = bt + C
$$

where $ C $ is the constant of integration. To determine the specific value of $ C $, we need additional information about the system, typically in the form of an **initial condition**.

As a further example, we can extend our previous example to:

$$
\frac{d^2x}{dt^2} = b,
$$

Notice this is a second-order ODE because the highest order derivative is second order. We can integrate once to get:

$$
\frac{dx}{dt} = bt + C_1
$$

and a second integration gives:

$$
x(t) = \frac{1}{2} bt^2 + C_1 t + C_2
$$

Each integration introduces a new unknown constant, $C_1$ and $C_2$, each of which require a unique piece of information about the system at some point in time to be fully defined. Generally this information described how the system we set up at the beginning of its motion at $t=0$, called **initial conditions**.


In summary, the order of a differential equation is crucial because it dictates:

- How many independent conditions are required for a unique solution. These are generally given as initial conditions describing how the system was set up at $t=0$. 
- The complexity of the solution process (higher-order equations often require special techniques).
- The physical behavior of the system. For example, we will learn that first-order equations describe simple exponential growth or decay, while second-order equations capture oscillatory and wave-like behavior.





### Homogeneous vs. Inhomogeneous Equations

A differential equation is classified as **homogeneous** if every term in the equation contains the function or one of its derivatives. In other words, a homogeneous equation has no standalone terms that are independent of the function being solved for.

For example, consider the second-order differential equation:

$$
\frac{d^2 x}{dt^2} + 4x = 0
$$

This equation is **homogeneous** because Every term in the equation contains either the function $ x(t) $ or one of its derivatives:

- The first term involves the second derivative of $ x(t) $.
- The second term involves $ x(t) $ itself.



On the other hand, a differential equation is **inhomogeneous** if it contains terms that do not involve the function or its derivatives. For example:

$$
\frac{d^2 x}{dt^2} + 4x = \cos(\omega t)
$$

Here, the additional term $ \cos(\omega t) $ does not involve $ x(t) $ or any of its derivatives. This makes the equation **inhomogeneous**.

In physics, inhomogeneous ODEs frequently arise in systems subject to external forces. The extra term generally represents an external input or disturbance rather than an inherent property of the system. Some common examples include:

- A **mass-spring system** with an external driving force:  
	
	$$
	m \frac{d^2 x}{dt^2} + kx = F_0 \cos(\omega t).
	$$
	
	The $ F_0 \cos(\omega t) $ term represents an external periodic force acting on the system, such as a child being pushed on a swing.
	
- An **electrical circuit** with an applied voltage source:  
	
	$$
	L \frac{d^2 Q}{dt^2} + R \frac{dQ}{dt} + \frac{1}{C} Q = V_0 \cos(\omega t).
	$$
	
	Here, $ V_0 \cos(\omega t) $ represents an external AC voltage applied to the circuit.


In each of these cases, the driving term forces the system into motion in a way that it would not naturally move on its own.





### Linear vs. Nonlinear Differential Equations

A differential equation is classified as **linear** if each term in the equation **only involves the function or one of its derivatives**, without any cross terms. This means that:

- The function $ x(t) $ and its derivatives appear **only to the first power**.
- The function and its derivatives are **not multiplied together**.


From this definition, we can see that a general linear ODE can be written in the form:

$$
a_n(t) \frac{d^n x}{dt^n} + a_{n-1}(t) \frac{d^{n-1} x}{dt^{n-1}} + \dots + a_1(t) \frac{dx}{dt} + a_0(t) x = f(t),
$$

where $ a_i(t) $ are functions of $ t $ and $ f(t) $ is an external forcing term.

To better understand this classification, consider the following examples of nonlinear differential equations. For example, the equation

$$
x \frac{dx}{dt} = 3
$$

is **nonlinear** because the function $ x(t) $ is multiplied by its derivative. Similarly, the equation

$$
\left( \frac{d^3 x}{dt^3} \right)^2 = \frac{dx}{dt} + 7
$$

is also **nonlinear** because the third derivative is squared.

Nonlinear ODEs are notoriously difficult to solve—if they have a solution at all. They often require numerical methods or qualitative approaches rather than direct algebraic solutions. Identifying whether an ODE is linear or nonlinear is crucial because it determines the solution strategies available. 

One key property of linear homogeneous ODEs is that they satisfy the **superposition principle**:
If $ x_1(t) $ and $ x_2(t) $ are solutions to a linear homogeneous ODE, then their sum,

$$
x_\text{combined}(t) = x_1(t) + x_2(t)
$$

is also a solution. We will prove this property when we discuss solution strategies for linear homogeneous ODEs.








### Constant vs. Variable Coefficient Equations

Another important distinction is whether the coefficients of the equation depend on time (or whatever independent variable is being used). If all coefficients are constant, the equation is a **constant-coefficient differential equation**, such as:

$$
\frac{d^2 x}{dt^2} + 5 \frac{dx}{dt} + 6x = 0
$$

These equations often admit solutions involving exponentials and trigonometric functions, making them easier to solve analytically.

On the other hand, if any coefficient depends on $ t $, the equation is a **variable-coefficient differential equation**, such as:

$$
t^2 \frac{d^2 x}{dt^2} + t \frac{dx}{dt} + x = 0
$$

Variable-coefficient equations are generally more difficult to solve, requiring techniques like power series solutions.



### Why These Classifications Matter

Understanding how differential equations are classified helps us choose the right solution technique. A first-order, linear ODE with constant coefficients can often be solved analytically with straightforward methods. A second-order, nonlinear, variable-coefficient ODE might require numerical integration or approximation techniques.

More importantly, recognizing these structures allows us to see connections between seemingly different physical systems. The same equation might describe electrical circuits, mechanical vibrations, and even population dynamics, just with different variable names.

Now that we have classified differential equations, let's move on to solving our first example: a simple first-order ODE describing flow rates of water.


{% capture ex %}
Below are several differential equations. For each, determine:

- Whether it is **linear** or **nonlinear**.
- The **order** of the equation.
- Whether it is **constant**- or **variable**-coefficient.
- Whether it is **homogeneous** or **inhomogeneous**.
- Whether it is an **ordinary** or a **partial** differential equation.


$$\frac{d^2y}{dx^2} + 3 \frac{dy}{dx} + 2y = 0$$

This **linear, second-order, constant-coefficient, homogeneous, ordinary differential equation**.

- **Linearity:** The function $ y(x) $ and its derivatives appear to the first power and are not multiplied together, so this is a **linear** equation.
- **Order:** This equation contains the second derivative $ d^2y/dx^2 $, so it is a **second-order** equation.
- **Coefficient:** None of the coefficients of $y(x)$ and its derivatives depend on $x$, making them **constant coefficients**. 
- **Homogeneity:** There are no standalone terms that do not include $ y $ or its derivatives, so the equation is **homogeneous**.
- **Type:** The only types of derivatives present are with respect to $x$. This is an **ordinary** differential equation.


---

$$\frac{d^3 x}{dt^3} + 4\left(\frac{dx}{dt} \right)^2 + x = \cos(t)$$

This **nonlinear, third-order, constant-coefficient, inhomogeneous, ordinary differential equation**.

- **Linearity:** The term $ \left(\frac{dx}{dt} \right)^2 $ contains a squared derivative, making this a **nonlinear** equation.
- **Order:** The highest derivative present is $ d^3x/dt^3 $, so this is a **third-order** equation.
- **Coefficient:** None of the coefficients of $x(t)$ and its derivatives depend on $t$, making them **constant coefficients**. 
- **Homogeneity:** The equation contains $ \cos(t) $, which does not include $ x(t) $ or its derivatives, so this equation is **inhomogeneous**.
- **Type:** The only types of derivatives present are with respect to $t$. This is an **ordinary** differential equation.


---

$$t \frac{d^2 y}{dt^2} + 2 \frac{dy}{dt} + 5y = 0$$

This **linear, second-order, variable-coefficient, homogeneous, ordinary differential equation**.

- **Linearity:** Each term contains either $ y $, its first derivative, or its second derivative, all raised to the first power. Since there are no products of $ y $ and its derivatives, the equation is **linear**.
- **Order:** The highest derivative present is $ d^2y/dt^2 $, making this a **second-order** equation.
- **Coefficient:** This coefficient of the second derivative is $t$, making this a **variable coefficients**. 
- **Homogeneity:** There is no standalone term independent of $ y $, meaning this equation is **homogeneous**.
- **Type:** The only types of derivatives present are with respect to $t$. This is an **ordinary** differential equation.



---


$$\frac{d^2 x}{dt^2} + x \frac{dx}{dt} + x^3 = 0$$

This **nonlinear, second-order, constant-coefficient, homogeneous, ordinary differential equation**.

- **Linearity:** The presence of the term $ x \frac{dx}{dt} $, where the function is multiplied by its derivative, makes this equation **nonlinear**.
- **Order:** The highest derivative is $ d^2x/dt^2 $, so this is a **second-order** equation.
- **Coefficient:** None of the coefficients of $x(t)$ and its derivatives depend on $t$, making them **constant coefficients**. 
- **Homogeneity:** Since every term contains $ x $ or its derivatives, the equation is **homogeneous**.
- **Type:** The only types of derivatives present are with respect to $t$. This is an **ordinary** differential equation.

---

$$\frac{dy}{dx} + 2y = e^x$$

This **linear, first-order, constant-coefficient, inhomogeneous, ordinary differential equation**.

- **Linearity:** The function $ y $ and its derivative appear to the first power and are not multiplied together, so the equation is **linear**.
- **Order:** The highest derivative present is $ dy/dx $, making this a **first-order** equation.
- **Coefficient:** None of the coefficients of $y(x)$ and its derivatives depend on $x$, making them **constant coefficients**. 
- **Homogeneity:** The term $ e^x $ does not contain $ y $ or its derivatives, meaning this equation is **inhomogeneous**.
- **Type:** The only types of derivatives present are with respect to $x$. This is an **ordinary** differential equation.


{% endcapture %}
{% include example.html content=ex %}













## First-Order Differential Equations – A Flow Rate Example

Now that we have established the different classifications of differential equations, let’s start discussing solution strategies. To keep things intuitive, we will begin with a physical system that most people have encountered at some point in their lives: filling a bucket with water.

Imagine you have a bucket with a small hole at the bottom, and you are filling it with water from a faucet. The rate at which water is added to the bucket, $ R_{\text{in}} $, is constant. At the same time, water is leaking out of the hole at a rate $ R_{\text{out}} $ that depends on how much water is currently in the bucket. This leakage follows a well-known physical principle: the more water in the bucket, the greater the pressure at the hole, and thus the faster the water leaks out.

Let $ h(t) $ represent the height of the water in the bucket as a function of time. The rate of change of $ h $ will depend on the amount of water entering and leaving the bucket. Water entering the bucket will try to fill it up, meaning the $ R_{\text{in}} $ term will lead to a positive $\frac{dh}{dt}$ if that is the only thing happening. That is, the height of the water in the bucket will rise due to $ R_{\text{in}} $. 

Similarly, $ R_{\text{out}} $ is going to try to decrease the height of the water in the bucket and will create a negative $\frac{dh}{dt}$ if there is no water being added to the bucket ($ R_{\text{in}} $). 

This means the rate at which the height of the water in the bucket will depdning on the in flow and out flow rates in the following manner:

$$
\frac{dh}{dt} = R_{\text{in}} - R_{\text{out}}
$$

From fluid mechanics, the outflow rate is proportional to the square root of the water height:

$$
R_{\text{out}} = k \sqrt{h}.
$$

Thus, our differential equation for the water level is:

$$
\frac{dh}{dt} = R_{\text{in}} - k\sqrt{h}.
$$

This equation tells us how the height of water in the bucket changes over time. It is a **first-order**, **nonlinear**, **constant-coefficient**, **inhomogeneous** **ordinary** differential equation because:

- It involves only the first derivative $ dh/dt $ (first-order).
- The term $ k \sqrt{h} $ makes it nonlinear (not just a sum of $ h $ and its derivatives).
- It has an external input $ R_{\text{in}} $ making it inhomogeneous.


Despite its nonlinearity, this equation is solvable using separation of variables, a technique we will introduce shortly.

### Steady-State/Equilibrium Solution

Before solving the equation formally, we can gain intuition by analyzing its **steady-state** behavior. The **steady-state** solution is what results when all of the derivatives of the function are set equal to zero. This is sometimes called the **equilibrium solution**. In this case, setting the first derivative equal to zero gives:

$$
\frac{dh}{dt} = 0 \quad \implies \quad R_{\text{in}} - k\sqrt{h_\text{eq}} = 0
$$

This means the system reaches equilibrium when the inflow equals the outflow:

$$
R_{\text{in}} = k\sqrt{h_{\text{eq}}}
$$

Solving for the equilibrium height, $ h_{\text{eq}} $, we find:

$$
h_{\text{eq}} = \left( \frac{R_{\text{in}}}{k} \right)^2.
$$

What this solution tells us is if we let the the water run long enough, and have a big enough bucket, the water level will stabilize at $ h_{\text{eq}} $, where the inflow and outflow will be balanced.

Finding the **steady-state solution** or **equilibrium solution** is one really useful technique for helping to understand and interpret the physical meaning of a differential equation.




## Separation of Variables and Solving First-Order ODEs

Now that we’ve seen examples of how first-order differential equations naturally arise in physics, let’s focus on how to solve them. One of the most straightforward and widely applicable techniques for solving a first-order ODE is called **separation of variables**. This method is particularly useful for differential equations in which all terms involving the function can be moved to one side of the equation, and all terms involving the working variable can be moved to the other.

### The General Method

Consider a first-order ordinary differential equation (ODE) of the form:

$$
\frac{dy}{dx} = f(x) g(y)
$$

Here, we are taking the derivative of $ y $ with respect to $ x $, meaning $ y $ must be a function of $ x $, i.e., $ y = y(x) $. As a result, we will refer to $ y $ as the **function** and $ x $ as the **working variable**. On the right-hand side, $ f(x) $ is a function that depends only on the working variable $ x $, while $ g(y) $ is a function that depends only on $ y $. 

A first-order ODE is said to be **separable** if it can be rearranged into the following form:

$$
\frac{dy}{g(y)} = f(x) \, dx
$$

This step involves dividing by $ g(y) $ and "multiplying" by $ dx $. Now, mathematicians tend to object when physicists and engineers perform this operation, as it is not entirely rigorous from a formal mathematical perspective. However, in practical applications, the types of functions we encounter almost always allow for this manipulation without violating any actual mathematical rules. Rather than performing a more abstract transformation, we take this convenient and effective approach—let's call it "physicist-friendly math."

Once the equation is rewritten in this separable form, both sides can be integrated independently:

$$
\int \frac{dy}{g(y)} = \int f(x) \, dx

$$
After performing these integrations (assuming they can be evaluated), the next step is to solve for $ y(x) $, provided that the equation allows for us to solve for $y(x)$. The resulting expression for $ y(x) $ is called the **general solution** to the differential equation.

We refer to this as the **general** solution because it describes all possible solutions for any system governed by the given differential equation. However, if we wish to describe a particular physical situation, we need additional conditions—such as initial values or boundary conditions—to determine the precise behavior of $ y(x) $. By applying these conditions, we can solve for any unknown constants introduced during integration, obtaining what is known as the **particular solution**. This particular solution is unique to the specific scenario defined by the given conditions.





{% capture ex %}
Example: Solving Radioactive Decay

Earlier, we wrote down the equation governing radioactive decay:

$$
\frac{dN}{dt} = -\lambda N
$$

This equation closely resembles the one we derived for the bucket problem, except that there is no inflow term ($ R_\text{in} $) in this case. Additionally, the outflow rate ($ R_\text{out} $) is written as $ \lambda N $, without any square root dependence. 


What is this equation telling us? It states that the rate at which particles decay away, $ \frac{dN}{dt} $, is proportional to the number of particles currently present, $ N(t) $. This makes perfect sense! The more radioactive particles present, the more decaying particles we expect to observe at any given moment.

Before solving the ODE, let's get the steady-state solution. Sett all derivatives equal to zero:

$$
0 = -\lambda N \quad \implies \quad  N_\text[Steady] = 0
$$

This means the number of radioactive particles in the sample will eventually go to zero, which makes sense considering they are all actively trying to decay.

Now, let’s solve this linear, first-order, constant-coefficient, homogeneous ODE. We begin by rewriting it in separable form:

$$
\frac{dN}{N} = -\lambda dt
$$

Next, we integrate both sides:

$$
\int \frac{dN}{N} = \int -\lambda dt
$$

Carrying out the integrations gives:

$$
\ln |N| = -\lambda t + C
$$

where $ C $ is the unknown constant of integration. 

To solve for $ N(t) $, we exponentiate both sides:

$$
N(t) = e^C e^{-\lambda t}
$$

This expression represents the **general solution** to the radioactive decay equation.

To determine a specific solution, we apply an initial condition:

$$
N(0) = N_0
$$

where $ N_0 $ represents the initial number of radioactive particles at $ t = 0 $. Substituting this into our general solution:

$$
N(0) = e^C e^{-\lambda (0)} = e^C e^{0} = e^C (1) = e^C
$$

Thus, we find:

$$
e^C = N_0
$$

Substituting this back into our general solution, we obtain the **particular solution**:

$$
N(t) = N_0 e^{-\lambda t}
$$

This equation describes exponential decay, a fundamental process seen not only in nuclear physics but also in chemical reactions, population dynamics, and many other physical systems.

{% endcapture %}
{% include example.html content=ex %}


{% capture ex %}
Example: Cooling of an Object (Newton’s Law of Cooling). 

Another classic application of separation of variables appears in heat transfer. Suppose an object, with temperature $ T(t) $, is placed in an environment that is maintained at a constant ambient temperature $ T_{\text{env}} $. Newton’s law of cooling states that the rate of change of the object's temperature is proportional to the difference between its temperature and the ambient temperature:

$$
\frac{dT}{dt} = -k\left(T - T_{\text{env}}\right)
$$

where $ k $ is a positive constant. The negative sign is present to force $\frac{dT}{dt}$ to be negative when $ T(t) > T_{\text{env}} $, meaning the the object decreases in temperature while losing heat to the surroundings.

Before solving this differential equation, let’s determine the steady-state solution. At equilibrium, the temperature no longer changes, so we set the derivative to zero:

$$
0 = -k\left(T_{\text{steady}} - T_{\text{env}}\right) \quad \implies \quad T_{\text{steady}} = T_{\text{env}}
$$

This confirms our intuition: eventually, the object's temperature will equal the ambient temperature.

Next, we rewrite the differential equation in separable form. Dividing both sides by $ T - T_{\text{env}} $ and multiplying by $ dt $ gives:

$$
\frac{dT}{T - T_{\text{env}}} = -k \, dt
$$

Now, integrate both sides:

$$
\int \frac{dT}{T - T_{\text{env}}} = \int -k \, dt
$$

Using the substitution $ u = T - T_{\text{env}} $ (with $ du = dT $), we obtain:

$$
\int \frac{du}{u} = \int -k dt \quad \implies \quad \ln|u| = - k t + C
$$

where $C$ is the unknown constant of integration. Subbing out $u$ leaves us with:

$$
\ln |T - T_{\text{env}}| = -kt + C
$$

Exponentiating both sides and solving for $T(t)$:

$$
T(t) = e^C e^{-kt} + T_{\text{env}}
$$

This is the general solution to this problem.

If we denote the initial temperature of the object by $ T_0 $ (so that $ T(0) = T_0 $), then:

$$  T(t=0) = e^C e^{-k(0)} + T_{\text{env}} =  e^C + T_{\text{env}} \quad \implies \quad T_0 = e^C + T_{\text{env}} \quad \implies \quad   e^C  = T_0  - T_{\text{env}}  $$

Thus, the particular solution can be given as:

$$
T(t) = \left( T_0 - T_{\text{env}} \right) e^{-kt} + T_{\text{env}}
$$

This result shows that the object's temperature approaches $ T_{\text{env}} $ exponentially, with the rate of cooling determined by $ k $. Notice, this agrees with the steady-state solution we got previously.

{% endcapture %}
{% include example.html content=ex %}



### Key Takeaways from Separation of Variables

Separation of variables is a powerful method that applies to many different types of physical systems:

- **Decay Processes:** Radioactive decay, discharging capacitors, and damping in mechanics all follow exponential decay laws.
- **Growth Processes:** Population growth, chemical reactions, and nuclear chain reactions use similar mathematics.
- **Thermal and Electrical Systems:** Newton’s law of cooling and RC circuits are directly modeled using separable differential equations.


This technique will serve as a foundational tool for solving many of the differential equations we encounter in physics and engineering. In the next lecture, we will expand our toolkit further by discussing more general solution methods for first-order differential equations.












## Interpreting Solutions: What Do They Tell Us?

At this point, we’ve developed a solid understanding of how to set up and solve first-order ordinary differential equations using separation of variables. But solutions to differential equations are more than just expressions with integrals and exponentials. They contain deep physical meaning—describing how a system evolves over time, how external forces shape its behavior, and what role initial conditions play. Let’s take a step back and examine what these solutions actually represent and why they matter.

### The Role of Initial Conditions

Unlike algebraic equations, where we solve for a single numerical value, a differential equation defines an entire family of solutions. This is because solving a first-order ODE typically involves integrating, which introduces an arbitrary constant $ C $. Mathematically, the presence of this constant reflects an important idea: differential equations describe *dynamical systems*, and to predict their exact behavior, we must specify an initial condition--what the system looked like when it was started.

Consider the classic example of radioactive decay, governed by:

$$
\frac{dN}{dt} = -\lambda N
$$

Solving this using separation of variables gives:

$$
N(t) = N_0 e^{-\lambda t}
$$

Here, $ N_0 $ is the initial condition—how much of the radioactive substance we had at $ t = 0 $. Without specifying $ N_0 $, we don’t have a unique solution! This tells us that differential equations don’t just encode how a system changes, but also demand information about where/how the system started.

This is a universal theme across physics:

- The motion of a falling object depends on its initial velocity.  
- The charge on a capacitor in an electrical circuit depends on its initial charge.  
- The population of a species in an ecosystem depends on the initial population size.  


Initial conditions select which specific solution from the general family actually describes the real-world situation.

### How Different Solutions Correspond to Different Physical Scenarios

The presence of arbitrary constants means that different initial conditions lead to different solutions. But sometimes, small differences in initial conditions lead to wildly different outcomes. This is a concept you might associate with chaos theory, but it also appears in many ordinary differential equations.

For example, consider Newton’s law of cooling:

$$
\frac{dT}{dt} = -k(T - T_{\text{env}})
$$

This describes how an object cools toward the temperature of its surroundings $ T_{\text{env}} $. Solving this gives:

$$
T(t) = T_{\text{env}} + (T_0 - T_{\text{env}}) e^{-kt}
$$

Here, the initial temperature $ T_0 $ plays a crucial role in determining the system's evolution. If $ T_0 $ is much higher than $ T_{\text{env}} $, the object takes longer to cool, whereas if it starts closer to $ T_{\text{env}} $, the cooling process takes less time to complete.

This underscores an important lesson:

- The solution to a differential equation isn’t just a function—it’s a model of how a system behaves over time.
- Small changes in initial conditions can result in vastly different outcomes, particularly in nonlinear systems.


In many real-world applications, understanding the range of possible solutions is just as important as solving for one particular case.

### When Analytical Solutions Aren’t Enough: The Need for Numerical Methods

In the examples we’ve covered, the differential equations were nice to us—they were separable and had straightforward analytical solutions. Unfortunately, the vast majority of differential equations encountered in physics do not yield to simple algebraic tricks.

For example, consider the logistic equation used in population dynamics:

$$
\frac{dN}{dt} = rN\left(1 - \frac{N}{K}\right)
$$

This equation describes a population growing under limited resources, where $ K $ is the carrying capacity. Unlike the simple exponential decay equation we saw earlier, this equation cannot be solved by elementary functions alone—it requires a more advanced technique known as the integrating factor method (which we will learn next lecture) or numerical approximations.

Similarly, the differential equations governing the motion of a pendulum in physics:

$$
\frac{d^2\theta}{dt^2} + \frac{g}{L} \sin \theta = 0
$$

cannot be solved exactly **unless** we assume small-angle approximations. This is why computational techniques, such as **Euler’s method, Runge-Kutta methods, and numerical integration**, become essential when working with real-world problems.

At this stage, it’s important to recognize that analytical solutions are great when they exist, but for complex systems, numerical solutions are often the only option. This being said, we will be focusing on analytic methods for the lectures that follow.















## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.



A water tank initially contains $ V_0 $ liters of water. Water is pumped into the tank at a **constant rate** $ R_{\text{in}} = \text{a constant} $. Simultaneously, water leaks out of the tank at a rate proportional to the square root of the current water volume, given by:

$$
R_{\text{out}} = k \sqrt{V(t)}
$$

where $ k $ is a positive constant and $ V(t) $ is the volume of water in the tank at time $ t $.


a) Write down the differential equation that governs the change in volume $ V(t) $ in the tank.  
	
>Hint: The rate of change of the volume ($\frac{dV}{dt}$) should have something to do with the rate of water being pored in and the rate of water leaving. This is *very* similar to the height of water in a bucket example we looked at previously.
	
b) Determine the steady-state volume $ V_{\text{Steady}} $ by setting the time derivative $ \frac{dV}{dt} $ to zero.  
	
c) Rewrite the differential equation in separable form and solve it to obtain the general solution for $ V(t) $.  
	
d) Given $ V(0) = V_0 $, find the particular solution for $ V(t) $.  
	
e) Explain the behavior of $ V(t) $ as $ t \to \infty $. What does the solution imply about the water volume in the tank over time? Does this agree with the steady-state solution you found?

