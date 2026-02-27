---
layout: default
title: Mathematical Methods - Lecture 12
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 12
---



# Lecture 12 – Solving First Order ODEs



## The Integrating Factor Method

In our previous lectures, we explored various methods for solving differential equations. We saw how separation of variables can elegantly solve many first-order ODEs when the equation neatly divides into one part that depend solely on the function and another part that depend solely on the working variable. However, not every linear, first-order ODE is easily separable. This is where the integrating factor method becomes an invaluable tool. 

The integrating factor method provides a systematic approach to solving linear, first-order, homogeneous/inhomogeneous, variable-coefficient ODEs in the standard form:

$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

where $P(x)$ and $Q(x)$ are functions of the working variable $x$. 

The integrating factor method poses the following question: Can we multiple the left hand side of the differential equation above by a function such that the left hand side turns into a product rule. We will clarify the specifics of this questions in the following section. But, for the time being, this carefully chosen function is called the **integrating factor**. Once we have rewritten the left hand side using the product rule, we should be able to simply integrate both sides directly and obtain the general solution.

This technique is not only mathematically elegant, but also highly practical. It is widely used in physics and engineering to analyze systems ranging from RC circuits and thermal processes to population dynamics and beyond. Mastering the integrating factor method equips you with a powerful tool to tackle a wide range of real-world problems modeled by differential equations.









### Derivation of the Integrating Factor Method

Consider a linear, first-order ODE written in **standard form**:

$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

Our goal is to solve for $y(x)$. The idea behind the integrating factor method is to multiply the entire equation by a function $\mu(x)$ (the *integrating factor*):

$$
\mu(x)\frac{dy}{dx} + \mu(x)P(x)y(x) = \mu(x)Q(x)
$$

where $\mu(x)$ chosen such that the left-hand side becomes the derivative of the product $\mu(x)y$:

$$
\mu(x)\frac{dy}{dx} + \mu(x) P(x)y = \frac{d}{dx} \big(\mu(x) y(x) \big)
$$

To find an equation for the integrating factor, let's begin by by expanding the right hand side:

$$ \mu(x)\frac{dy}{dx} + \mu(x)P(x)y(x) =  \frac{d\mu}{dx} y(x) + \mu(x) \frac{dy}{dx}  $$

We can cancel out the $\mu(x)\frac{dy}{dx}$ terms on both sides and divide out $y(x)$ to leaves us with:

$$ \mu(x)P(x) =  \frac{d\mu}{dx}  $$

This is a separable first-order ODE! Using separation of variables to solve for the integrating factor gives us:

$$ 
\begin{aligned}
\mu(x)P(x) &=  \frac{d\mu}{dx} \\
 \frac{d\mu}{\mu} &= P(x) \, dx  \\
\ln\vert \mu(x)\vert  &= \int P(x) \, dx
\end{aligned}
$$

where we have ignored the integration constant here as it will be redundant considering the integration constant we will also get when we integrate to get $y(x)$. Solving for $\mu(x)$ gives us:

$$
\mu(x) = e^{\int P(x) \, dx}
$$

This is what $\mu(x)$ needs to be to allow the product rule simplification we demanded be possible to simplify the differential equation. 

Speaking of, using that product rule simplification allows us to write the original ODE as: 

$$ \frac{d}{dx} \left( \mu(x) y(x) \right) = \mu(x)Q(x) $$

which we can solve by integrating both sides with respect to $x$:

$$ \int \frac{d}{dx} \left( \mu(x) y(x) \right) \, dx  = \int \mu(x)Q(x) \, dx $$

we can use the fundamental theorem of calculus to use the integration to cancel out the derivative up to a additive constant $C$:

$$ \mu(x) y(x) = \int \mu(x)Q(x) \, dx + C $$

and finally divide $\mu(x)$ to the other side to get the general solution:

$$ y(x) = \frac{1}{\mu(x)} \left( \int \mu(x)Q(x) \, dx + C \right) $$

where, as a reminder: 

$$ \mu(x) = e^{\int P(x) \, dx}  $$



{% capture ex %}
If you have a first-order ODE of the following standard form: 

$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

Then the general solution will be:

$$ y(x) = \frac{1}{\mu(x)} \left( \int \mu(x)Q(x) \, dx + C \right) $$

where $\mu(x)$,the integrating factor, is:

$$ \mu(x) = e^{\int P(x) \, dx}  $$
{% endcapture %}
{% include example.html content=ex %}












## Worked Examples for the Integrating Factor Method


### Example 1: An RC Circuit

Consider an RC circuit where the charge $ Q(t) $ on the capacitor satisfies the differential equation derived from Kirchhoff's loop rule:

$$
\frac{dQ}{dt} + \frac{1}{RC} Q = \frac{V_0}{R}
$$

with the initial condition $ Q(0) = 0 $. Here, $ R $ is the resistance, $ C $ is the capacitance, and $ V_0 $ is the applied constant voltage.

**Step 0: Understand the Problem, Predict the Outcome, and Find the Steady-State Solution**

In this problem, the capacitor is initially uncharged ($ Q(0) = 0 $) and is then connected to a voltage source $ V_0 $. Physically, we expect the capacitor to charge over time until its voltage matches $ V_0 $. From introductory physics, we know that the maximum charge the capacitor can hold is given by:

$$
Q_{\text{max}} = V_0 C
$$

Initially, the capacitor charges rapidly because there is little voltage across it to oppose the current. As charge accumulates, the capacitor’s voltage rises, slowing the charging process until it asymptotically approaches $ Q_{\text{max}} = V_0 C $.

This conceptual analysis is crucial as it helps us anticipate that the solution will have an exponential approach to equilibrium rather than oscillatory behavior, and it reinforces our understanding of the physical process. 

Lastly, it is generally helpful to get the steady-state solution. Setting the derivatives equal to zero gives:

$$
0 + \frac{1}{RC} Q_\text{Steady} = \frac{V_0}{R} \quad\implies\quad Q_\text{Steady} = V_0 C
$$

which agrees with what we conceptually expected to see for this particular problem.

This step is not mandatory, but is highly suggested before simply solving the ODE. 

**Step 1: Write in Standard Form**

We begin with the differential equation for an RC circuit:

$$
\frac{dQ}{dt} + \frac{1}{RC} Q = \frac{V_0}{R}
$$

and recognize it to be written in standard form:

$$
\frac{dQ}{dt} + P(t) Q = f(t)
$$

In this equation, the coefficient of $ Q $ is $ P(t) = \frac{1}{RC} $ and the inhomogeneous forcing term is $ f(t) = \frac{V_0}{R} $—we can't call it $Q(t)$ since that is being used to represent the charge on the capacitor.

**Step 2: Determine the Integrating Factor**

The integrating factor $ \mu(t) $ is defined by:

$$
\mu(t) = e^{\int P(t)\, dt}
$$

Substituting $ P(t) = \frac{1}{RC} $ into the integral, we obtain:

$$
\mu(t) = e^{\int \tfrac{1}{RC}\, dt} = e^{\tfrac{t}{RC}}
$$

Using this integrating factor, we multiply the entire differential equation by $ e^{t/(RC)} $, which will allow us to write the left-hand side as the derivative of the product $ \mu(t) Q(t) $. 

**Step 3: Apply the General Solution**

The general solution to this problem can be given as:

$$ Q(t) = \frac{1}{\mu(t)} \left( \int \mu(t)f(t) \, dt + A \right) $$

where $A$ is our unknown constant of integration (we switched from $C$ to $A$ since $C$ is being used for capacitance in this problem). 

First, let's substitute in the integrating factor $\mu(x)$ and the function for $f(t)$ we identified previously: 

$$ Q(t) = \frac{1}{e^{\tfrac{t}{RC}}} \left( \int \left(e^{\tfrac{t}{RC}} \right) \left(\frac{V_0}{R}\right) \, dt + A \right) $$

Simplifying and pulling constants out of the integral gives:

$$ Q(t) = e^{\tfrac{-t}{RC}} \left( \frac{V_0}{R} \int e^{\tfrac{t}{RC}} \, dt + A \right) $$

Performing the integration:

$$ Q(t) = e^{\tfrac{-t}{RC}} \left( \frac{V_0}{R} \left( RC e^{\tfrac{t}{RC}} \right) + A \right) $$

and simplifying:

$$ Q(t) =  V_0 C  + A e^{\tfrac{-t}{RC}} $$

This is the general solution to this problem. 

**Step 4: Apply Initial Conditions**

We were told the initial charge on the capacitor was $Q(0) = 0$. Applying this gives:

$$ 
\begin{aligned}
Q(0) &= 0 \\
V_0 C  + A e^{\tfrac{-(0)}{RC}} &= 0 \\
V_0 C  + A &= 0 \\
A &= - V_0 C    
\end{aligned}
$$

Putting this into the general solution gives us a particular solution of:

$$ Q(t) =  V_0 C  - V_0 C  e^{\tfrac{-t}{RC}} $$

which can be simplified to:

$$ Q(t) =  V_0 C \left( 1  - e^{\tfrac{-t}{RC}}\right) $$

**Step 5: Interpretation of Result and Verification**

The solution

$$ Q(t) =  V_0 C \left( 1  - e^{\tfrac{-t}{RC}}\right) $$

describes how the capacitor charges over time. As $ t \to \infty $, $ e^{\tfrac{-t}{RC}} \to 0 $ and $ Q(t) $ approaches $ V_0 C $, which was what we expected to find from both out conceptual and steady-state investigations. This confirms the physical intuition that the capacitor's voltage will eventually match the voltage source $ V_0 $.

This example demonstrates the systematic approach of the integrating factor method and shows how to interpret the solution in a real-world context.









### Example 2: Newton's Law of Cooling

Consider an object with temperature $ T(t) $ placed in an environment with a constant ambient temperature $ T_{\text{env}} $. Newton’s law of cooling states that the rate at which the object's temperature changes is proportional to the difference between its temperature and the ambient temperature:

$$
\frac{dT}{dt} = k(T_{\text{env}}-T(t))
$$

with the initial condition $ T(0) = T_0 $, and $ k $ is a positive constant.

**Step 0: Understand the Problem, Predict the Outcome, and Find the Steady-State Solution**

In this scenario, the object starts at an initial temperature $ T_0 $ and is placed in an environment maintained at $ T_{\text{env}} $. We expect, over time, the object's temperature will approach $ T_{\text{env}} $. For instance, if $ T_0 > T_{\text{env}} $, the object's temperature will decrease to $T_\text{env}$; if $ T_0 < T_{\text{env}} $, the object's temperature will increase to $T_\text{env}$. From basic physics, and everyday reasoning, we know the final equilibrium temperature should be $ T_{\text{env}} $.

Checking this logic by finding the steady-state solution, we can take all derivatives to zero:

$$
0 = k(T_{\text{env}}-T_{\text{Steady}}) \quad\implies\quad T_{\text{Steady}} = T_{\text{env}}
$$

This agrees with our previous reasoning. 

**Step 1: Write in Standard Form**

The differential equation can be written in standard form as:

$$
\frac{dT}{dt} + kT = kT_{\text{env}}
$$

where the coefficient of $ T $ is $ P(t) = k $ and the inhomogeneous forcing term is $ Q(t) = kT_{\text{env}} $.

**Step 2: Determine the Integrating Factor**

The integrating factor $ \mu(t) $ can be found to be:

$$
\mu(t) = e^{\int P(t)\, dt} = e^{\int k\, dt} = e^{kt}
$$

**Step 3: Apply the General Solution**

Taking the general solution: 

$$ T(t) = \frac{1}{\mu(t)} \left( \int \mu(t)Q(t) \, dt + C \right) $$

and putting the integrating factor and $Q(t)$ gives:

$$ T(t) = \frac{1}{e^{kt}} \left( \int \left(e^{kt}\right) \left( kT_{\text{env}} \right) \, dt + C \right) $$

which we can simplify:

$$ T(t) = e^{-kt} kT_{\text{env}} \left( \int e^{kt}\, dt + C \right) $$


From here we can complete the integration:

$$ T(t) = e^{-kt} kT_{\text{env}} \left( \frac{1}{k} e^{kt} + C \right) $$

leaving us with a general solution of:

$$ T(t) = T_{\text{env}} \left(  1 + kC e^{-kt}  \right) $$



**Step 4: Apply Initial Conditions**
At $ t = 0 $, we are told $ T(0) = T_0 $, so:

$$
\begin{aligned}
T(0) &= T_0 \\
T_{\text{env}} \left(  1 + kC e^{-k(0)}  \right) &= T_0 \\
T_{\text{env}} \left(  1 + kC \right) &= T_0   \\
k C &= \frac{T_0}{T_{\text{env}}} - 1
\end{aligned}
$$

Thus,

$$
T(t) = T_{\text{env}} \left(  1 + \left(\frac{T_0}{T_{\text{env}}} - 1 \right)  e^{-kt}  \right) 
$$

which we can simplify by distributing $T_{\text{env}}$ through both parentheses:

$$
T(t) = T_{\text{env}}  +  \left( T_0 - T_{\text{env}}  \right)  e^{-kt} 
$$


**Step 5: Interpretation of the Result and Verification**

The solution

$$
T(t) = T_{\text{env}} + \left( T_0 - T_{\text{env}} \right)e^{-kt}
$$

describes an exponential approach of the object's temperature toward the ambient temperature $ T_{\text{env}} $. As $ t \to \infty $, $ e^{-kt} \to 0 $ and $ T(t) $ approaches $ T_{\text{env}} $, which confirms our expectation that the object will eventually reach thermal equilibrium with its surroundings.






### Example 3: A Mixing Problem in a Tank

Consider a tank that initially contains $ V $ liters of water and $ A_0 $ kilograms of salt. Salt water with a constant concentration $ C_{\text{in}} $ (in kg/L) flows into the tank at a rate $ R $ (L/min), and the well-mixed solution inside the tank exits at the same rate $ R $, keeping the volume $ V $ of the solution in the tank constant. Let $ A(t) $ represent the amount of salt (in kilograms) in the tank at time $ t $.

**Step 0: Understand the Problem and Predict the Outcome**

In this problem, the tank is initially loaded with $ A_0 $ kg of salt. As salt water enters at a concentration of $ C_{\text{in}} $ and the solution exits at the same rate, we expect that over time the concentration of salt in the tank will approach the same value as the inflow concentration $ C_{\text{in}} $. Since the tank’s volume remains constant, the steady-state amount of salt will be:

$$
A_{\text{eq}} = V\, C_{\text{in}}
$$

This is what we should expect for the long term state of whatever solution we get from the ODE that models this particular situation.

**Step 1: Write in Standard Form**

The rate of change of salt in the tank is the difference between the salt inflow and outflow. The inflow rate of salt will be the flow rate timed the input concentration:

$$ R_\text{in} = R\, C_{\text{in}} $$

The out flow rate will be the current concentration of the salt in the tank, given by $A(t)/V$ times the out flow rate. Since the volume of water in the tank is constant we know the outflow rate of salt water will be $R$. This allows us to write the outflow rate as:

$$ R_\text{out} = R\, \left( \frac{A(t)}{V} \right) = \frac{R}{V} \, A(t)  $$

We can build an ODE for this by recognizing that the amount of salt in the tank will increase by the inflow rate and decrease by the outflow rate:

$$
\frac{dA}{dt} = R_\text{in} - R_\text{out}
$$

Putting what we have found give the following ODE:

$$
\frac{dA}{dt} = R\, C_{\text{in}} - \frac{R}{V}\, A(t)
$$

Rearranging this equation into standard linear form:

$$
\frac{dA}{dt} + \frac{R}{V} A = R\, C_{\text{in}}
$$

We can identify $ P(t) = \frac{R}{V} $ and the forcing term is $ Q(t) = R\, C_{\text{in}} $.

**Step 2: Determine the Integrating Factor**

The integrating factor $ \mu(t) $ can be found to be:

$$
\mu(t) = e^{\int P(t)\, dt} = e^{\int \frac{R}{V}\, dt} = e^{\frac{R}{V} t}
$$

**Step 3: Apply the General Solution**

The general solution can be written in the following form:

$$ A(t) = \frac{1}{\mu(t)} \left( \int \mu(t)Q(t) \, dt + C \right) $$

Putting the integrating factor and $Q(t)$ gives:

$$ A(t) = \frac{1}{e^{\frac{R}{V} t}} \left( \int \left( e^{\frac{R}{V} t} \right) \left( R\, C_{\text{in}}\right) \, dt + C \right) $$

which we can simplify:

$$ A(t) = e^{-\frac{R}{V} t} \, R\, C_{\text{in}} \,  \left( \int e^{\frac{R}{V} t} \, dt + C \right) $$

Integrating:

$$ A(t) = e^{-\frac{R}{V} t} \, R\, C_{\text{in}} \,  \left( \frac{V}{R} e^{\frac{R}{V} t} + C \right) $$

and simplifying gives us the general solution for this problem:

$$ A(t) = V\, C_{\text{in}}  + R\, C_{\text{in}} \, C e^{-\frac{R}{V} t}$$

**Step 4: Apply the Initial Condition**

We are given the initial amount of salt in the tank as $ A(0) = A_0 $. Using this gives:

$$
\begin{aligned}
A(0) &= V\, C_{\text{in}}  + R\, C_{\text{in}} \, C e^{-\frac{R}{V} (0)} \\
 A_0 &= V\, C_{\text{in}}  + R\, C_{\text{in}} \, C \\
A_0 - V\, C_{\text{in}}  &= R\, C_{\text{in}} \, C
\end{aligned}
$$

Putting this into the general solution:

$$ A(t) = V\, C_{\text{in}}  + \left( A_0 - V\, C_{\text{in}} \right) e^{-\frac{R}{V} t}$$

gives us the particular solution for this problem. 

**Step 5: Interpretation of the Result and Verification**


We can do a quick check of the particular solution by verifying it goes to the steady-state value we conceptually expected: $ t \to \infty $, $ e^{-\frac{R}{V} t} \to 0 $ and $ A(t) $ approaches $ V\, C_{\text{in}} $, as expected.

This solution confirms our prediction: over time, the amount of salt in the tank stabilizes at $ A_{\text{eq}} = V\, C_{\text{in}} $. The term $ \left( A_0 - V\, C_{\text{in}} \right) e^{-\frac{R}{V} t} $ describes the transient behavior of the system; that is, the rate at which the system approaches equilibrium.









## Autonomous ODEs: Reducible Second-Order ODE

Occasionally you will be faced with an ODE that appears to be second-order, but can be handled as if it were a first-order ODE. Equations like this are called **autonomous** ODEs, meaning they can be reduced to a lower order through a simple change of variables. 

This kind of situation will generally, but not always, occur in the following manner. Consider the standard form for a second-order ODE:

$$ A(x) \frac{d^2 y(x)}{dx^2} + B(x) \frac{d y(x)}{dx} + C(x) = 0 $$

suppose, instead of these functions depending **explicitly** on $x$ (meaning there is an $x$ in one of the functions $A(x)$, $B(x)$, and $C(x)$) supposed they only depended **implicitly** on $x$ through $y(x)$ or one of its derivatives. This would mean we can write the standard for something like this:

$$ A(y', y) y'' + B(y', y) y' + C(y', y) = 0 $$

Here the $x$ dependent is hidden inside the function $y$ and its derivatives: $\frac{dy}{dx} = y'$, $\frac{d^2y}{dx^2} = y''$, and so on. 

If this is the case, then we can propose using $y$ as the working variable as instead of $x$ since it doesn't even appear in the differential equation. This change of variables would require us to update the derivatives using chain rule and what not. 

To help this along, and to help keep our heads screwed on straight, we can make the following substitution:

$$ \text{Let: } v = y' $$

where we say $v$ is a function of $y$, not $x$. That is $v = v(y)$.  

Now we would like to rewrite all of the old derivatives with respect to $x$ to new derivatives being taken with respect to $y$. We can do this through the application of the chain-rule:

$$
y' = v \quad \implies \quad y'' = \frac{dv}{dx} = \frac{dv}{dy}\frac{dy}{dx} = \frac{dv}{dy} \ y' = \frac{dv}{dy} \ v = v \ \frac{dv}{dy}
$$

These adjustments allow us to rewrite the above ODE in the following manner: 

$$ A(v, y) \  v \ \frac{dv}{dy}  + B(v, y) v + C(v, y) = 0 $$

However, notice that $B(v, y) v + C(v, y) = D(v, y)$ is just a function depending on $v$ and $y$. So, we can introduce a new function to replace that combination, $D(v, y) =  B(v, y) v + C(v, y)$. This gives us:

$$ A(v, y) \ v \ \frac{dv}{dy}  + D(v, y) = 0 $$

We can now use first-order ODE solution techniques to solve this problem. 

This is all well and good, but this general sketch of the process may not have made any sense or it may have made this process seem more difficult than it actually is in practice. Let's look at an example of this process in action to help settle any potential confusion. 


{% capture ex %}
Consider the following ODE 

$$
y'' = \frac{(y')^2}{y}
$$

Since the equation does not explicitly depend on $ x $, we will reduce its order by making the substitution suggested in the general outline:

$$
y' = v \quad \implies \quad y'' = v \ \frac{dv}{dy}
$$

Substituting into the original equation gives:

$$
v \frac{dv}{dy} = \frac{v^2}{y}
$$

Assuming $ v \neq 0 $, we can use separation of variables to solve this problem. Dividing both sides by $ v^2 $ and multiplying the $dy$ to the right hand side we get:

$$
\frac{dv}{v} = \frac{dy}{y}
$$

We can integrate both sides to get:

$$
\int \frac{dv}{v} = \int \frac{dy}{y} \quad \implies \quad \ln\vert v\vert  = \ln\vert y\vert  + C
$$

where $ C $ is the unknown constant of integration. Exponentiating both sides:

$$
\vert v\vert  = e^C  \vert y \vert  \quad \Longrightarrow \quad v = A y
$$

where $ A  $ is a new arbitrary constant to replace the constant exponential $e^C$ and to absorb any potential minus signs floating around to take care of the absolute valued nature of the solution.

We have sound the general solution for $v(y)$, but not the solution for $y(x)$ that the problem was initially set up to find. Recalling that $ v = y' $, the equation becomes:

$$
y' = A y
$$

This is a standard first-order linear ODE whose general solution can be found using separation of variables again. We get a general solution:

$$
y(x) = B e^{Ax}
$$

where $ B $ is yet another unknown constant of integration. 

Notice we have two unknown constants of integration $A$ and $B$. This makes sense since the ODE we solved was second-order, meaning we had to integrate twice to fully solve for $y(x)$. For every integration we take we get an unknown constant, thus two constants should appear in our solution. To get a particular solution we would need to be given two condition about the problem to find values for $A$ and $B$. 
{% endcapture %}
{% include example.html content=ex %}















## Exact Differential Equations

With what we have learned so far, we are ready to tackle a first-order partial differential equation (PDE). This may seem intimidating at first, but we will be focusing on a specific form of first-order PDE called an exact differential equation.

Exact differential equations form a special class of first-order ODE/PDEs (depending on how you write it) that can be expressed in the form

$$
M(x,y) \ dx + N(x,y) \ dy = 0
$$

Let's assume there is a function $F(x,y)$ that exists such that:

$$
\frac{\partial F}{\partial x} = M(x,y) \quad \text{and} \quad \frac{\partial F}{\partial y} = N(x,y)
$$

If this is possible, we call $F(x,y)$ the potential function for this problem. (Yes, this is related to potential energies and the like, if you were wondering.)

Since $F$ is a function of both $x$ and $y$, its total differential will be given as:

$$
dF = \frac{\partial F}{\partial x} \ dx + \frac{\partial F}{\partial y} \ dy 
$$

We can then substitute in the relations between the partial derivatives of $F$ and the $M$ and $N$ functions to get:

$$
dF = M(x,y) \ dx + N(x,y) \ dy
$$

But, this is the original PDE we were given, except it was was equal to zero. This means:

$$
dF = 0
$$

The only way for a differential of a function to be zero is if the function was actually a constant. Thus, the general solution of the differential equation can be written as:

$$
F(x,y) = C
$$

where $C$ is an arbitrary constant.

### How do we know the differential equation is exact or now?

To determine whether the equation

$$
M(x,y)\,dx + N(x,y)\,dy = 0
$$

is **exact**, meaning it represents the exact differential of a function of $x$ and $y$, we can perform the following check:

$$
\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}
$$

If this condition holds, then there exists a function $F(x,y)$ such that $dF = 0$ is equivalent to the given equation.



{% capture ex %}
Consider the differential equation:

$$
(2xy + 3)\,dx + (x^2 + 4y)\,dy = 0
$$

We can identify the functions $M$ and $N$ as:

$$
M(x,y) = 2xy + 3 \qquad N(x,y) = x^2 + 4y
$$

and use that check to see if this represents an exact differential equation:

$$
\frac{\partial M}{\partial y} = 2x \qquad \frac{\partial N}{\partial x} = 2x
$$

Since $\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$, the equation is exact.

#### Finding the Potential Function $F(x,y)$

To find $F(x,y)$, we first integrate $M(x,y)$ with respect to $x$. But, why?

Recall, we said 

$$M(x,y) = \frac{\partial F}{\partial x}$$ 

So, integrating this with respect to $x$ will remove the $x$ partial derivative. The only thing we have to remember is that the constant of integration will be an unknown function of $y$ instead of a simple unknown constant. This is because the partial derivative with respect to $x$ of a function that only depends on $y$ will be zero. This gives us:

$$
F(x,y) = \int (2xy + 3) \, dx = x^2y + 3x + g(y)
$$

where $g(y)$ is that arbitrary function of $y$ only.

Next, we can partial differentiate this form for $F(x,y)$ with respect to $y$ because this should be the function $N$, recall we said 

$$N(x,y) = \frac{\partial F}{\partial y}$$ 

This gives:

$$
N(x,y) = \frac{\partial F}{\partial y} \quad\implies\quad x^2 + 4y = x^2 + g'(y)
$$ 

Thus,

$$
g'(y) = 4y
$$

and integrating with respect to $y$ gives:

$$
g(y) = 2y^2 + A
$$

where $A$ is a constant of integration, not an unknown function.

Putting this $g(y)$ into what we found for $F(x,y)$ gives the potential function as:

$$
F(x,y) = x^2y + 3x + 2y^2 + A
$$

#### General Solution

Now, remember we said that $F(x,y)$ needed to be a constant to actually be the solution to the exact differential equation. Thus, the general solution to the differential equation is given by:

$$
F(x,y) = x^2y + 3x + 2y^2 = C
$$

where we let $C$ absorb the constant $A$.
{% endcapture %}
{% include example.html content=ex %}












## Additional Worked Examples for First-Order ODEs

Let's consolidate our understanding of first-order differential equations by working through several examples. These examples demonstrate both the separation of variables and the integrating factor methods, and they illustrate how these techniques are applied in various physics, engineering, and other applied contexts.

### Example 1: Exponential Growth (Separation of Variables)

Consider a simple model of population growth where the rate of change of the population $ P(t) $ is proportional to the current population:

$$
\frac{dP}{dt} = rP
$$

where $ r $ is the growth rate.

We can solve this problem by first recognizing that we can use separation of variables to solve this problem. Separating the variables gives: 

$$
\frac{dP}{P} = r\, dt
$$
  
Integrating both sides gives us

$$
\ln \vert P \vert  = rt + C
$$

which we can exponentiate and solve for $P(t)$:

$$
P(t) = e^{rt+C} = e^C e^{rt}
$$

This is our general solution for this problem.

We can find the particular solution by applying an initial condition of $P(0) = P_0$, where $P_0$ represents the initial population. Applying this give:

$$
P(0) = e^C e^{r(0)} = e^C \implies P_0 = e^C
$$

Putting this into the general solution gives us a particular solution of:

$$
P(t) = P_0 e^{rt}
$$

This is the classic exponential growth model.




### Example 2: Motion with Linear Drag (Integrating Factor)

Consider an object of mass $ m $ falling under gravity while experiencing a drag force proportional to its velocity. The drag force is given by

$$
F_{\text{drag}} = - b v
$$

where $ b $ is the drag coefficient and $ v(t) $ is the object's velocity. Notice this is written in a one-dimensional manner, meaning the drag force points in the **opposite direction** as the velocity of the object.  According to Newton's second law, the net force can be related to the acceleration, and from a free-body diagram we can convince ourselves that the forces acting on the object will be gravity and the drag force:

$$
\begin{aligned}
F_\text{net} &= F_\text{grav} + F_\text{drag} \\
m \frac{d^2 x}{dt^2} &= -mg - b \frac{dx}{dt}\\
\frac{d^2 x}{dt^2} &= -g - \frac{b}{m}  \frac{dx}{dt}
\end{aligned}
$$

Notice this is an autonomous ODE (the working variable  $t$ does not appear in the problem at all). In this case if we let: 

$$
v = \frac{dx}{dt} \quad\implies\quad \frac{d^2 x}{dt^2} = \frac{dv}{dt}
$$

Putting this into the original ODE gives:

$$
\frac{dv}{dt} = -g - \frac{b}{m}  v
$$

Now, writing this ODE in standard form 

$$
\frac{dv}{dt} +  \frac{b}{m}  v = -g
$$

and we can identify the following functions 

$$
P(t) = \frac{b}{m} \quad \text{and} \quad Q(t) = -g
$$

The integrating factor is defined as

$$
\mu(t) = e^{\int P(t) \, dt} = e^{\int \frac{b}{m}\, dt} = e^{\frac{b}{m} t}
$$

Here we would normally substitute stuff into the general solution when using an integrating factor. But what do you do if you forgot what the genera solution for this situation looked like? In this event, we remember that the integrating factor was designed with one goal in mind: rewrite the left-hand side of the ODE in standard form as a product rule of 

$$ \frac{d}{dt} \left( \mu(t) v(t)\right)  $$

Remembering this, we can multiply our ODE by $ e^{\frac{b}{m} t} $ to get:

$$
e^{\frac{b}{m} t}\frac{dv}{dt} + \frac{b}{m}e^{\frac{b}{m} t} v = -g\,e^{\frac{b}{m} t}
$$

Recognize that the left-hand side is the derivative of $ e^{\frac{b}{m} t} v(t) $, which it better be since we designed the integrating factor to do just that, we rewrite get:

$$
\frac{d}{dt}\left( e^{\frac{b}{m} t} v(t) \right) = -g\, e^{\frac{b}{m} t}
$$

Integrating both sides with respect to $ t $:

$$
e^{\frac{b}{m} t} v(t) = - \frac{g m}{b} e^{\frac{b}{m} t} + C
$$

where $ C $ is the constant of integration.

Finally, we can solve for $v(t)$ be dividing both sides by $ e^{\frac{b}{m} t} $:

$$
v(t) = -\frac{g m}{b} + C\,e^{-\frac{b}{m} t}
$$

Let's assume the initial velocity is given by $ v(0) = v_0 $, then:

$$
v(0) = - \frac{g m}{b} + C = v_0 \quad \implies \quad C = v_0 + \frac{g m}{b}
$$

After simplifying, the particular solution is:

$$
v(t) = -\frac{g m}{b} + \left( v_0 + \frac{g m}{b} \right) e^{-\frac{b}{m} t}
$$

We could integrate this with respect to time again to get the equation for the position. We will leave that to the reader to do, if they are interested. 

We can check this answer in a couple of different ways. First, what is the steady-state solution? We can get this by setting all of the derivatives in the original ODE to zero:

$$ \frac{dv}{dt} = 0  \quad \implies \quad 0  = -g - \tfrac{b}{m}  v_\text{Steady} \quad \implies \quad  v_\text{Steady} = - \frac{gm}{b} $$

Why is this negative? Because the object will be falling down! 

Let's compare this to what we get then we take our particular solution and let $t \rightarrow \infty$. In this limit we will get $  e^{-\frac{b}{m} t} \rightarrow 0$. This gives:

$$ v(t \rightarrow \infty) =  -\frac{g m}{b} + \left( v_0 + \frac{g m}{b} \right) (0) \quad \implies \quad  v(t \rightarrow \infty) = -\frac{g m}{b} $$

which agrees with our steady state solution. This offers some level of confidence that we solved the problem properly. 








### Advanced Application: The SIS Model in Epidemiology

The SIS (Susceptible-Infected-Susceptible) model is widely used in epidemiology to describe the spread of diseases in which individuals can become infected, recover, and then return to the susceptible population. This would be a good model for a seasonal flu they everyone is susceptible every year. A typical formulation of the SIS model is given by:

$$
\frac{dI}{dt} = \beta I (N - I) - \gamma I
$$

where:

- $ I(t) $ is the number of infected individuals at time $ t $,
- $ N $ is the total (constant) population,
- $ \beta $ is the transmission rate, and
- $ \gamma $ is the recovery rate.


We can rearrange this equation into a more standard form as:

$$
\frac{dI}{dt} = (\beta N - \gamma)  I  - \beta I^2
$$

Notice that this equation is **nonlinear** due to the product $ \beta I^2 $. As a result, the integrating factor method cannot be directly applied to solve the full SIS model, and using of separation of variables will lead us to a rather unpleasant integration.

In many practical situations we make a simplifying assumtion that the nonlinear terms in the ODE are so small that we can safely ignore them. For instance, during the early stages of an outbreak when the number of infected individuals is small relative to the total population ($ I \ll N $), we can approximate the equation by assuming $ I^2 \approx 0 $. Under this approximation, the SIS model simplifies to:

$$
\frac{dI}{dt} \approx (\beta N - \gamma)  I 
$$

This is now a linear, first-order differential equation:

$$
\frac{dI}{dt} = (\beta N - \gamma) I
$$

which can be solved by either separation of variables or the integrating factor method. For example, using separation of variables:

$$
\frac{dI}{I} = (\beta N - \gamma) \, dt
$$

integrating both sides yields:

$$
\ln  \vert I \vert  = (\beta N - \gamma) t + C
$$

and exponentiating gives:

$$
I(t) = e^{C} e^{(\beta N - \gamma)t}
$$

If we call the initial number of people infected $I(0) = I_0$, we can get:

$$
I(0) = I_0 = e^{C} 
$$

This gives use the particular solution of:

$$
I(t) = I_0 e^{(\beta N - \gamma)t}
$$

To help explain this solution we can make the following rearrangements:

$$
I(t) = I_0 e^{\gamma (\tfrac{\beta N}{\gamma} - 1)t}
$$

$$
I(t) = I_0 e^{\gamma (R_0 - 1)t}
$$

where $R_0 = \frac{\beta N}{\gamma}$ is called the reproduction number. If $R_0 < 1$ that means less than one person is infected by every 1 infected person and the disease will eventually die out. However, if $R_0 > 1$ that means more than one person is infected by every 1 infected person and the disease will grow into a pandemic.

 
We can be a bit more careful by examining the steady state behavior of the original ODE:

$$
\frac{dI}{dt} = (\beta N - \gamma)  I  - \beta I^2
$$

Setting the derivative to zero gives:

$$
\begin{aligned}
\frac{dI}{dt} &= 0   \\ 
(\beta N - \gamma)  I  - \beta I^2 & = 0 \\
 ( (\beta N - \gamma)  - \beta I) \ I&= 0
\end{aligned}
$$

$$
I_\text{Steady} = \frac{\gamma}{\beta} (R_0 - 1) \quad\text{or}\quad I_\text{Steady} = 0
$$

Obviously, the $I_\text{Steady} = 0$ is the ideal solution since it means the infection will eventually die out and there will be no infections left. 

Of concern is the other solution: $ I_\text{Steady} = \frac{\gamma}{\beta} (R_0 - 1)$. Notice, if $R_0 < 1$, then this steady state solution will be negative, which makes no sense for the number of infected people. If $R_0 > 1$ then this steady state solution will be positive, which means there is a solution where the number of infected people never goes to zero. This is referred to as an endemic, a disease that never truly goes away. 

In conclusion, while the full SIS model is nonlinear and typically requires numerical or qualitative methods for analysis. In this case the steady state solution helps to shed light on possible outcomes of the model and how to possibly mitigate potentially devastating outcomes. Additionally, the linear approximation of the differential equation near the disease-free equilibrium (this process is called *linearization*) can be solved using our standard techniques and not only provides insight into the early dynamics of the disease spread but also helps us understand concepts like the basic reproduction number $ R_0 = \frac{\beta N}{\gamma} $, which determines whether an infection will die out or lead to an epidemic.

For those who are interested, the solution to the full ODE can be found using a Bernoulli substitution method. We will learn this method next lecture.





















## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


A fish population in a lake grows exponentially with a per capita growth rate $ r $ but is subject to a constant harvest of $ H $ fish per unit time. The dynamics of the population $ P(t) $ are modeled by the differential equation

$$
\frac{dP}{dt} = rP - H
$$

with the initial condition

$$
P(0) = P_0
$$

**Tasks:**

(a) Write the differential equation in the standard linear form.  

(b) Determine the integrating factor and use it to solve for the general solution $ P(t) $.  

(c) Find the equilibrium population $ P_{\text{eq}} $ by setting $\frac{dP}{dt} = 0$.  

(d) Discuss the long-term behavior of $ P(t) $ in terms of $ r $ and $ H $. In particular, explain what happens as $ t \to \infty $ and how changes in the harvest rate $ H $ affect the equilibrium population.



**Hint:**  
Recall that for a first-order linear ODE in standard form  

$$
\frac{dP}{dt} + \left(-r\right)P(t) = -H
$$

Find the integrating factor and use it to transform the equation into one that can be integrated directly.  

Do not forget to apply your initial conditions!



















