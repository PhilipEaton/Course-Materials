---
layout: default
title: Mathematical Methods - Lecture 16
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 16
---

# Lecture 16 – Inhomogeneous Second-Order Linear ODEs - Variation of Parameters

## Introduction

So far, we’ve developed a toolkit for solving second-order linear ODEs, starting with homogeneous equations. For constant-coefficient cases, we used the standard approach of guessing exponential solutions and solving the resulting characteristic equation. When dealing with variable coefficients, we introduced the method of reduction of order, which helps find a second solution when one solution is already known. 

When shifting our focus to inhomogeneous equations, we’ve saw that solving them requires both a homogeneous solution and a particular solution. The method of **undetermined coefficients** gave us an efficient way to find particular solutions when the forcing function was a simple polynomial, exponential, or sine/cosine function. However, it has a major limitation: it only works when the forcing function fits into a specific set of standard forms, like the following:


| Driving Term $f(x)$ | Guess for Particular Solution $y_p(x)$ |
|---|---|
| $P_n(x) = 2x^n + 3x^{n-1} + \cdots - 9x + 2$ | $Q_n(x) = a_nx^n + a_{n-1}x^{n-1} + \cdots + a_1x + a_0$ |
| $e^{ax}$ | $Ae^{ax}$ |
| $\cos(bx)$ or $\sin(bx)$ | $A\cos(bx) + B\sin(bx)$ |
| $e^{ax}\cos(bx)$ or $e^{ax}\sin(bx)$ | $e^{ax}\left(A\cos(bx) + B\sin(bx)\right)$ |
| $P_n(x)e^{ax}$ | $e^{ax}Q_n(x)$ |
| $P_n(x)\cos(bx)$ or $P_n(x)\sin(bx)$ | $Q_n(x)\cos(bx) + R_n(x)\sin(bx)$ |

**Note:** $P_n(x)$, $Q_n(x)$, and $R_n(x)$ are polynomials of degree $n$.

**Note:** If your initial guess for $y_p(x)$ is already a solution to the homogeneous equation, multiply your guess by $x$ (or a higher power of $x$ if needed) to ensure linear independence.
 

What happens if the forcing function is something more complicated, like $ \ln(x) $ or $ \frac{1}{x} \sin(x) $? In these cases, undetermined coefficients won’t work, and we need a more general approach. 

This is where **variation of parameters** comes in. Instead of guessing a fixed form for the particular solution, this method used the homogeneous solution, representing the response of the system, and lets the unknown coefficients themselves vary dynamically instead of forcing them to be constant. This added flexibility allows us to tackle a much wider range of problems. The trade-off, however, is that variation of parameters involves integration, which can sometimes be tedious. But the added effort is worth it, as this method gives us a powerful tool for handling a broad class of differential equations.


##  Method of Variation of Parameters

This approach is more general than the method of undetermined coefficients and involves using the known linearly independent homogeneous solutions to derive a particular solution of the form:

$$
y_p(x) = u_1(x)y_1(x) + u_2(x)y_2(x)
$$

where $ u_1(x) $ and $ u_2(x) $ are functions that will need to be determined using the inhomogeneous ODE.

As an example of how this method works, consider this constant-coefficient, inhomogeneous, second-order ODE:

$$
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = f(x)
$$

where $f(x)$ is the driving (or the inhomogeneous) term. Let's suppose $ y_1(x) $ and $ y_2(x) $ are two linearly independent solutions of the homogeneous ODE, meaning both satisfy:

$$
\begin{aligned}
a\frac{d^2y_1}{dx^2} + b\frac{dy_1}{dx} + c\,y_1 &= 0 \\
a\frac{d^2y_2}{dx^2} + b\frac{dy_2}{dx} + c\,y_2 &= 0 
\end{aligned}
$$

and we can use the Wronskian to check if $ y_1(x) $ and $ y_2(x) $ linearly independent solutions.

For this method we will use a particular solution of the form:

$$
y_p(x) = u_1(x)y_1(x) + u_2(x)y_2(x)
$$

where $ u_1(x) $ and $ u_2(x) $ are functions to be determined. Calculating the first derivative of the particular solution gives:

$$
y_p'(x) = u_1'(x)y_1(x) + u_1(x)y_1'(x) + u_2'(x)y_2(x) + u_2(x)y_2'(x)
$$

Before calculating the second derivative, we **impose an auxiliary condition**:

$$
u_1'(x)y_1(x) + u_2'(x)y_2(x) = 0
$$

which simplifies the first derivative of the particular solution to: 

$$
y_p'(x) = u_1(x)y_1'(x) + u_2(x)y_2'(x)
$$

The auxiliary condition is chosen to simplify the first derivative of the particular solution by removing the first derivatives of the unknown functions $u_1'(x)$ and $u_2'(x)$ from experiencing another derivative. That is, implementing this auxiliary condition will result in the second derivative of the particular solution only depending on the first derivative of the unknown functions, thus reducing the problem from one second-order equation to 2 first-order equations. To see this, take the second derivative of the particular solution with the auxiliary condition implemented:

$$
y_p''(x) = u_1'(x)y_1'(x) + u_1(x)y_1''(x) + u_2'(x)y_2'(x) + u_2(x)y_2''(x)
$$

Notice, no second derivatives of $u_1$ and $u_2$ are present. 

Putting these derivatives into the inhomogeneous ODE gives:

$$
\begin{aligned}
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y &= f(x) \\[1.5ex]
a\Bigl( u_1'(x)y_1'(x) + u_1(x)y_1''(x) + u_2'(x)y_2'(x) + u_2(x)y_2''(x)  \Bigr)\hspace{1.5cm} \\
+\, b\Bigl( u_1(x)y_1'(x) + u_2(x)y_2'(x) \Bigr) + c\,\Bigl( u_1(x)y_1(x) + u_2(x)y_2(x) \Bigr) & = f(x) \hspace{3cm}  \\[1.5ex]
\end{aligned}
$$

We can group these on $u_1(x)$, $u_2{x}$ and their derivatives to get:

$$ a u_1'(x) y_1'(x) + u_1(x) \underbrace{\Big(a y_1''(x) + b y_1'(x) + c y_1(x)\Big)}_{=0} + u_2(x) \underbrace{\Big(a y_1''(x) + b y_1'(x) + c y_1(x)\Big)}_{=0} + a u_2'(x)y_2'(x) = f(x) $$

where we can make the observation that the homogeneous ODE pops out as the coefficient of $u_1(x)$ and $u_2(x)$, which will be zero as indicated. After dividing by $a$, we have the following equation:

$$u_1'(x) y_1'(x) + u_2'(x)y_2'(x) = \tfrac{1}{a} f(x) $$

Combining this result with the auxiliary equation, we have the following system of two equations:

$$
\begin{aligned}
u_1'(x)y_1(x) + u_2'(x)y_2(x) &= 0 \\
u_1'(x)y_1'(x) + u_2'(x)y_2'(x) &= \frac{f(x)}{a}
\end{aligned}
$$

We can solve these for for $ u_1'(x) $ and $ u_2'(x) $. First, let's use the top equation to find a relation for $ u_1'(x) $ in terms of $ u_2'(x) $:

$$ u_1'(x)y_1(x) + u_2'(x)y_2(x) = 0 \qquad \implies \qquad u_1'(x) =  - \frac{y_2(x)}{y_1(x)} \, \, u_2'(x) $$

Putting this into the second equation allows us to get an equation for $ u_2'(x) $:

$$
\begin{aligned}
u_1'(x)y_1'(x) + u_2'(x)y_2'(x) &= \frac{f(x)}{a} \\[1.5ex] 
\Big(- \frac{y_2(x)}{y_1(x)} \, \, u_2'(x)\Big) y_1'(x) + u_2'(x)y_2'(x) &= \frac{f(x)}{a} \\[1.5ex] 
\Big(- \frac{y_2(x)y_1'(x)}{y_1(x)}   + y_2'(x) \Big) \, u_2'(x) &= \frac{f(x)}{a} \\[1.5ex] 
\Big(- \frac{y_2(x)y_1'(x)}{y_1(x)}   + \frac{y_1(x) y_2'(x)}{y_1(x)} \Big) \, u_2'(x) &= \frac{f(x)}{a} \\[1.5ex] 
\Big(\frac{y_1(x) y_2'(x) - y_2(x)y_1'(x)}{y_1(x)} \Big) \, u_2'(x) &= \frac{f(x)}{a} \\[1.5ex] 
u_2'(x) &= \frac{f(x)}{a} \,\, \frac{y_1(x)}{y_1(x) y_2'(x) - y_2(x)y_1'(x)} \\[1.5ex] 
\end{aligned}
$$

which also gives:

$$ u_1'(x) =  - \frac{y_2(x)}{y_1(x)} \, \, u_2'(x) \qquad \implies \qquad  u_1'(x) =  - \frac{f(x)}{a} \,\, \frac{y_2(x)}{y_1(x) y_2'(x) - y_2(x)y_1'(x)} $$

Integrating gives $ u_1(x) $ and $ u_2(x) $. 

$$
\begin{aligned}
u_1(x) &=  - \frac{1}{a}  \int  \,\, \frac{f(x) \, y_2(x)}{y_1(x) y_2'(x) - y_2(x)y_1'(x)} \, dx\\
u_2(x) &=  \frac{1}{a}  \int  \,\, \frac{f(x) \, y_1(x)}{y_1(x) y_2'(x) - y_2(x)y_1'(x)} \, dx
\end{aligned}
$$

This allows us to write the particular solution in the following manner:

$$
\begin{aligned}
y_p(x) &= u_1(x)y_1(x) + u_2(x)y_2(x) \\[1.5ex]
&= y_1(x) \Big(- \frac{1}{a}  \int  \,\, \frac{f(x) \, y_2(x)}{y_1(x) y_2'(x) - y_2(x)y_1'(x)} \, dx\Big) + y_2(x)\Big(\frac{1}{a}  \int  \,\, \frac{f(x) \, y_1(x)}{y_1(x) y_2'(x) - y_2(x)y_1'(x)} \, dx\Big) \\[1.5ex]
y_p(x) &= - \frac{y_1(x) }{a} \Big( \int \frac{f(x) \, y_2(x)}{y_1(x) y_2'(x) - y_2(x)y_1'(x)} \, dx\Big) + \frac{y_2(x)}{a}  \Big( \int \frac{f(x) \, y_1(x)}{y_1(x) y_2'(x) - y_2(x)y_1'(x)} \, dx\Big)
\end{aligned}
$$

where you have to be careful to complete the indefinite integrals before multiplying in the coefficients.  

Notice, the denominator of the integrals is the Wronskian between the two functions:

$$ W(y_1,y_2)(x) = \begin{vmatrix}
	y_1 & y_2 \\ y_1' & y_2'
\end{vmatrix} = y_1 y_2' - y_2 y_1' $$

This allows us to write the particular solution in the form:

$$
y_p(x) = - \frac{y_1(x) }{a} \Big( \int \frac{f(x) \, y_2(x)}{W(y_1,y_2)(x)} \, dx\Big) + \frac{y_2(x)}{a}  \Big( \int \frac{f(x) \, y_1(x)}{W(y_1,y_2)(x)} \, dx\Big)
$$

This formula tells us that once we find two independent homogeneous solutions $y_1(x)$ and $y_2(x)$, we can compute the particular solution by evaluating these integrals. Notice, the Wronskian of $y_1(x)$ and $y_2(x)$ must be nonzero for this to work since it appears in the denominator. This implies $y_1(x)$ and $y_2(x)$ need to be linearly independent for variation of parameters to be applied.

To see how this works in practice, let’s apply variation of parameters to some specific ODEs.



{% capture ex %}
Let's use variation of parameters to solve the following differential equation:

$$
y'' - 3y' + 2y = e^x
$$

First we need to get the homogeneous solutions. That means we have to solve the homogeneous ODE:

$$
y'' - 3y' + 2y = 0
$$

We assume a solution of the form $ y_h = e^{rx} $, which leads to the characteristic equation:

$$
r^2 - 3r + 2 = 0
$$

Factoring and solving for $r$ gives:

$$
(r - 1)(r - 2) = 0 \qquad \implies \qquad r = 1 \quad \text{and} \quad r = 2
$$

This results in the following general homogeneous solution:

$$
y_h(x) = C_1 e^x + C_2 e^{2x}
$$

Armed with the homogeneous solutions $ y_1(x) = e^x $ and $ y_2(x) = e^{2x} $, assuming they are linearly independent, we can move into the variation of parameters process. We begin by looking for a particular solution of the form:

$$
y_p(x) = u_1(x) e^x + u_2(x) e^{2x}
$$

Taking the first derivative:

$$
y_p'(x) = u_1'(x) e^x + u_1(x) e^x + u_2'(x) e^{2x} + 2u_2(x) e^{2x}
$$

and imposing the **auxiliary condition**:

$$
u_1'(x) e^x + u_2'(x) e^{2x} = 0
$$

helps simplify the first derivative to:

$$
y_p'(x) = u_1(x) e^x + 2u_2(x) e^{2x}
$$

Differentiating again to get the second derivative leaves us with:

$$
y_p''(x) = u_1'(x) e^x + u_1(x) e^x + 2u_2'(x) e^{2x} + 4u_2(x) e^{2x}
$$

Substituting $ y_p''(x) $, $ y_p'(x) $, and $ y_p(x) $ into the inhomogeneous ODE:

$$
y'' - 3y' + 2y = e^x
$$

gives us:

$$
u_1'(x) e^x + 2u_2'(x) e^{2x} = e^x
$$

after fully simplifying (you are welcome to verify the algebra on your own).

We now have the following system of equations:

$$
\begin{aligned}
	u_1'(x) e^x + u_2'(x) e^{2x} &= 0 \\
	u_1'(x) e^x + 2u_2'(x) e^{2x} &= e^x
\end{aligned}
$$

We can solve for $u_2'(x)$ by subtracting the two equations (first minus the second) cancels out the first terms in each equation to get:

$$
u_2'(x) e^{2x} = e^x
$$

Dividing by $ e^{2x} $, assuming it is not equal to zero:

$$
u_2'(x) = e^{-x}
$$

give us an equation for $u_2'(x)$.  Substituting this into the first equation in our system of equations:

$$
u_1'(x) e^x - e^x = 0
$$

$$
u_1'(x) = 1
$$

Integrating both results gives:

$$
u_2(x) = -e^{-x} \qquad \text{and} \qquad u_1(x) = x
$$

Since $ y_p(x) = u_1(x) e^x + u_2(x) e^{2x} $, we substitute $ u_1(x) $ and $ u_2(x) $:

$$
y_p(x) = x e^x - e^{x}
$$

Notice the second term is actually one of the homogeneous solutions. As a result we can drop that term, leaving us with a particular solution of:

$$
y_p(x) = x e^x
$$

The general solution to the differential equation is:

$$
y(x) = C_1 e^x + C_2 e^{2x} + x e^x
$$

{% endcapture %}
{% include example.html content=ex %}







{% capture ex %}
Let's use variation of parameters to solve the following differential equation:

$$
y'' - 4y' + 4y = \ln(x), \quad x > 0
$$

First, we solve the homogeneous equation:

$$
y'' - 4y' + 4y = 0
$$

to get the homogeneous solutions. Assuming a solution of the form $ y_h = e^{rx} $, leads to the following characteristic equation and roots:

$$
r^2 - 4r + 4 = 0 \quad\implies\quad (r - 2)(r - 2) = 0 \quad\implies\quad r = 2 \quad\text{and}\quad r = 2
$$

Notice this is a repeated root of $ r = 2 $, so the homogeneous solution will be of the form:

$$
y_h(x) = C_1 e^{2x} + C_2 x e^{2x}
$$

From this homogeneous solution, we can identify out two linearly independent solutions, which can be verified using the Wronskian, are:

$$
y_1(x) = e^{2x} \quad\text{and}\quad y_2(x) = x e^{2x}
$$

Variation of parameters directs us to look for a particular solution in the form:

$$
y_p(x) = u_1(x) \, e^{2x} + u_2(x) \, xe^{2x}
$$

Differentiating:

$$
y_p'(x) = u_1'(x) e^{2x} + 2 u_1(x) e^{2x} + u_2'(x) x e^{2x} + u_2(x) e^{2x} + 2u_2(x) x e^{2x}
$$
	
and applying the **auxiliary condition**:

$$
u_1'(x) e^{2x} + u_2'(x) x e^{2x} = 0
$$

leaves us with the following for the first derivative:

$$
y_p'(x) = 2 u_1(x) e^{2x} + u_2(x) e^{2x} + 2u_2(x) x e^{2x}
$$

Differentiating again to get the second derivative:

$$
y_p''(x) = 2 u_1'(x) e^{2x} + 4 u_1(x) e^{2x} + u_2'(x) e^{2x} + 2 u_2(x) e^{2x} + 2u_2'(x) x e^{2x} + 2u_2(x) e^{2x} + 4 u_2(x) x e^{2x}
$$

Finally, substituting the particular solution and its derivatives into the inhomogeneous equation:

$$
y_p'' - 4y_p' + 4y_p = \ln(x)
$$

we obtain:

$$
\begin{aligned}
	y_p'' - 4y_p' + 4y_p &= \ln(x) \\[1.15ex]
	\Big(2 u_1'(x) e^{2x} + 4 u_1(x) e^{2x} + u_2'(x) e^{2x} + 2 u_2(x) e^{2x} + 2u_2'(x) x e^{2x} + 2u_2(x) e^{2x} + 4 u_2(x) x e^{2x}\Big) \\
		- 4\Big(2 u_1(x) e^{2x} + u_2(x) e^{2x} + 2u_2(x) x e^{2x}\Big) + 4\Big(u_1(x) \, e^{2x} + u_2(x) \, xe^{2x}\Big) &= \ln(x) \\[1.5ex]
	2 u_1'(x) e^{2x} + 4 u_1(x) e^{2x} + u_2'(x) e^{2x} + 2 u_2(x) e^{2x} + 2u_2'(x) x e^{2x} + 2u_2(x) e^{2x} + 4 u_2(x) x e^{2x} \\
	-8 u_1(x) e^{2x} -4 u_2(x) e^{2x} - 8 u_2(x) x e^{2x} + 4 u_1(x) \, e^{2x} + 4 u_2(x) \, xe^{2x} &= \ln(x) \\[1.5ex]
	2 u_1'(x)  e^{2x} + u_1(x) \Big( 4e^{2x} -8 e^{2x} + 4 e^{2x} \Big)  + u_2'(x) \Big( e^{2x} + 2 x e^{2x}  \Big)   \\
		+ u_2(x) \Big( 2e^{2x} + 2 e^{2x} + 4 x e^{2x} -4 e^{2x} - 8  x e^{2x}  + 4 xe^{2x} \Big)  &= \ln(x) \\[1.5ex]
	2 u_1'(x)  e^{2x} + u_2'(x) ( 2x + 1 ) e^{2x}  &= \ln(x) 
\end{aligned}
$$

So, we have the following set of equations: 

$$
\begin{aligned}
	u_1'(x) e^{2x} + u_2'(x) x e^{2x} &= 0 \\
	2 u_1'(x)  e^{2x} + u_2'(x) ( 2x + 1 ) e^{2x}  &= \ln(x)
\end{aligned}
$$

Notice the second equation can be rearranged to get:

$$
2 \underbrace{\left( u_1'(x)  e^{2x} + u_2'(x) xe^{2x} \right)}_{=0} + u_2'(x) e^{2x}  = \ln(x)
$$

where the first term is just the auxiliary condition, and so equates to zero and removes the whole term. This leaves us with:

$$
 u_2'(x) e^{2x}  = \ln(x) \quad\implies\quad  u_2'(x)   = e^{-2x} \ln(x)
$$

From the first of the two equations we can find:

$$
u_1'(x) = - u_2'(x) x
$$

Substituting this in what we found for $ u_2'(x)$:

$$
u_1'(x) = - x e^{-2x} \ln(x)
$$

Now we can integrate these to get:

$$
u_2(x) = \int e^{-2x} \ln(x) dx
$$

$$
u_1(x) = -\int x e^{-2x} \ln(x) dx
$$

These integrals may not have elementary solutions, but they can be left in integral form or evaluated numerically if needed.

The particular solution can be given as:

$$
y_p(x) = - e^{2x} \Big( \int x e^{-2x} \ln(x) dx \Big) + x e^{2x}  \Big( \int e^{-2x} \ln(x) dx \Big)
$$

and the general solution to the differential equation is:

$$
y(x) = C_1 e^{2x} + C_2 x e^{2x} + - e^{2x} \Big( \int x e^{-2x} \ln(x) dx \Big) + x e^{2x}  \Big( \int e^{-2x} \ln(x) dx \Big)
$$
{% endcapture %}
{% include example.html content=ex %}





##  Interpretation and Geometric Insight

We have derived, and seen a couple of examples, the variation of parameters method. To reiterate, this method allows us to construct a particular solution to an inhomogeneous differential equation using the homogeneous solutions, assuming they are linearly independent. But why does this work? The key idea here is that we have allowed the coefficients of the homogeneous solution to vary dynamically. 

In the homogeneous solution to a second-order ODE, we construct solutions using constant coefficients:

$$
y_h(x) = C_1 y_1(x) + C_2 y_2(x)
$$

where $ y_1(x) $ and $ y_2(x) $ are linearly independent solutions to the homogeneous equation. The **fixed** coefficients $ C_1 $ and $ C_2 $ are selected to meet a specific set of initial conditions.

When a system is *forced* by a nonzero $ f(x) $, we know the homogeneous solutions alone are insufficient. The trick is instead of using **fixed** coefficients $ C_1 $ and $ C_2 $, we upgrade these coefficients to become **dynamic** functions of $ x $:

$$
y_p(x) = u_1(x)y_1(x) + u_2(x)y_2(x)
$$

By allowing the coefficients $ u_1(x) $ and $ u_2(x) $ to vary, we can dynamically adjusts the coefficients of $ y_1(x) $ and $ y_2(x) $ to meet the conditions given by the driving term $f(x)$ at each point $ x $. This is similar to meeting the initial conditions, but instead we are meeting a set of conditions for each point in the solution space by allowing the coefficients to adjust as needed. This process can be visualized as constructing a curved surface in the solution space described by $ u_1(x) $ and $ u_2(x) $, rather than staying along a fixed linear path described by $ C_1 $ and $ C_2 $. What do we mean by this?

A second-order differential equation defines a two-dimensional solution space, where any solution can be expressed as a combination of two linearly independent functions $ y_1(x) $ and $ y_2(x) $.  In the homogeneous case, solutions remain on a **fixed plane** spanned by $ y_1(x) $ and $ y_2(x) $. In the inhomogeneous case, the forcing function $ f(x) $ distorts this plane, causing solutions to **curve away** from purely homogeneous solutions.

One way to visualize this is by considering the trajectory of a particle in two-dimensional space where $ y_1(x) $ and $ y_2(x) $ form a **coordinate basis**, like unit vectors in a plane. In the homogeneous case, a solution moves along a straight-line combination of these two basis functions dictated by $ C_1 $ and $ C_2 $. In the inhomogeneous case, the forcing function continuously **perturbs** the trajectory, altering how the solution progresses through the solution space.

Since the homogeneous solutions $ y_1(x) $ and $ y_2(x) $ define a coordinate system for the solution space, their **Wronskian** determines whether they provide a valid linearly independent basis. If the Wronskian $ W(y_1, y_2) $ is nonzero, then we know $ y_1(x) $ and $ y_2(x) $ are linearly independent and will span the entire space of solutions, ensuring that any function $ f(x) $ can be accommodated through some choice of $ u_1(x) $ and $ u_2(x) $.

This reinforces why variation of parameters works: it leverages the full space of solutions to find a particular solution that dynamically adjusts to match $ f(x) $.














##  Additional Examples


##  Example 1

Now, consider the differential equation:

$$
y'' - y' - 2y = e^{x}\sin(x)
$$

The homogeneous part is the same as before:

$$
y'' - y' - 2y = 0
$$

with characteristic equation:

$$
r^2 - r - 2 = 0 = (r - 2) (r + 1)
$$

yielding roots $ r = 2 $ and $ r = -1 $. Thus,

$$
y_h(x) = C_1 e^{2x} + C_2 e^{-x}
$$


Now we will apply variation of parameters and assume a particular solution of the form:

$$
y_p(x) = u_1(x)e^{2x} + u_2(x)e^{-x}
$$

where $ u_1(x) $ and $ u_2(x) $ are functions to be determined. We impose the auxiliary condition:

$$
u_1'(x)e^{2x} + u_2'(x)e^{-x} = 0
$$

We have put this guess into the ODE, so we differentiate once to get:

$$
y_p'(x) = u_1'(x)e^{2x} + 2u_1(x)e^{2x} + u_2'(x)e^{-x} - u_2(x)e^{-x}
$$

Because we assume the auxiliary condition $ u_1'(x)e^{2x} + u_2'(x)e^{-x} = 0 $, this simplifies to:

$$
y_p'(x) = 2u_1(x)e^{2x} - u_2(x)e^{-x}
$$

Differentiate again:

$$
y_p''(x) = 2u_1'(x)e^{2x} + 4u_1(x)e^{2x} - u_2'(x)e^{-x} + u_2(x)e^{-x}
$$

Substituting $ y_p $, $ y_p' $, and $ y_p'' $ into the ODE:

$$
\begin{aligned}
	&\Big(2u_1'(x)e^{2x} + 4u_1(x)e^{2x} - u_2'(x)e^{-x} + u_2(x)e^{-x}\Big) \\
	&\quad - \Big(2u_1(x)e^{2x} - u_2(x)e^{-x}\Big) - 2\Big(u_1(x)e^{2x} + u_2(x)e^{-x}\Big) = e^{x}\sin(x)
\end{aligned}
$$

Notice that the terms involving $ u_1(x) $ and $ u_2(x) $ cancel (you can show this upon simplifying), leaving us with:

$$
2u_1'(x)e^{2x} - u_2'(x)e^{-x} = e^{x}\sin(x)
$$

We have the following system of equations:

$$
\begin{aligned}
	u_1'(x)e^{2x} + u_2'(x)e^{-x} &= 0 \\
	2u_1'(x)e^{2x} - u_2'(x)e^{-x} &= e^{x}\sin(x)
\end{aligned}
$$


Adding the two equations will cancel out the $u_2'(x)e^{-x}$ terms to give: 

$$ 3u_1'(x)e^{2x} = e^{x}\sin(x) \quad\implies\quad u_1'(x) = \frac{1}{3} e^{-x}\sin(x) $$

and subtracting the bottom equation from 2 times the top equation will cancel our the first terms to give:

$$ -3u_2'(x)e^{-x} = e^{x}\sin(x) \quad\implies\quad u_2'(x) = -\frac{1}{3} e^{2x}\sin(x) $$

Integrate each expression:

$$
u_1(x) = \int \frac{1}{3} e^{-x}\sin(x)\,dx
$$

$$
u_2(x) = \int -\frac{1}{3} e^{2x}\sin(x)\,dx
$$

(These integrals can be evaluated using integration by parts. For brevity, you may leave your answers in integral form or use a computerized integration system if the integration become too cumbersome.)

This gives the particular solution as:

$$ y_p(x) = e^{2x} \Big(\int \frac{1}{3} e^{-x}\sin(x)\,dx\Big) + e^{-x} \Big(\int -\frac{1}{3} e^{2x}\sin(x)\,dx \Big)  $$


and the general solution to the inhomogeneous ODE is:

$$
y(x) = y_h(x) + y_p(x) = C_1e^{2x} + C_2e^{-x} + y_p(x)
$$



##  Example 2: Solving an Inhomogeneous Equation Using Cramer's Rule

Consider the differential equation:

$$
x^2 y'' - 3x y' + 4y = \ln(x)
$$

This is a Cauchy-Euler equation, but we will solve it using variation of parameters.

The associated homogeneous equation is:

$$
x^2 y'' - 3x y' + 4y = 0
$$

We assume a solution of the form $ y = x^r $, which leads to the characteristic equation:

$$
r(r-1) - 3r + 4 = 0 \quad \implies \quad r^2 - 4r + 4 = 0  \quad \implies \quad (r - 2)(r - 2) = 0
$$

Thus, we have a repeated root of $ r = 2 $, which means the two linearly independent solutions to the homogeneous equation are:

$$
y_1(x) = x^2 \quad\text{and}\quad y_2(x) = x^2 \ln(x)
$$

where the second solution can be found using reduction of order, for example.

From there we assume a particular solution of the form:

$$
y_p(x) =  u_1(x) x^2 + u_2(x) x^2 \ln(x)
$$

Differentiating:

$$
y_p'(x) = u_1'(x) x^2 + 2 u_1(x) x + u_2'(x) x^2 \ln(x) + 2 u_2(x) x \ln(x) + u_2(x) x
$$

Applying the **auxiliary condition**:

$$
u_1'(x) x^2 + u_2'(x) x^2 \ln x = 0
$$

simplifies this first derivative to:

$$
y_p'(x) = 2 u_1(x) x + 2 u_2(x) x \ln(x) + u_2(x) x
$$


Differentiating again:

$$
y_p''(x) = 2 u_1'(x) x + 2 u_1(x) + 2 u_2'(x) x \ln(x) + 2 u_2(x) \ln(x) + 2 u_2(x) + u_2'(x) x + u_2(x)
$$

Substituting into the original equation:

$$
\begin{aligned}
	x^2 \Big(2 u_1'(x) x + 2 u_1(x) + 2 u_2'(x) x \ln(x) + 2 u_2(x) \ln(x) + 2 u_2(x) + u_2'(x) x + u_2(x)\Big) \\
	- 3x \Big(2 u_1(x) x + 2 u_2(x) x \ln(x) + u_2(x) x\Big) + 4\Big(u_1(x) x^2 + u_2(x) x^2 \ln(x) \Big) &= \ln(x) \\[1.5ex]
	2 u_1'(x) x^3 + 2 x^2 u_1(x) + 2 u_2'(x) x^3 \ln(x) + 2 u_2(x) x^2 \ln(x) + 2 x^2 u_2(x) + u_2'(x) x^3 + u_2(x) x^2  \\
	- 6  u_1(x) x^2 - 6  u_2(x) x^2 \ln(x) - 3 u_2(x) x^2 + 4 u_1(x) x^2 + 4 u_2(x) x^2 \ln(x)  &= \ln(x) \\[1.5ex]
	2 u_1'(x) x^3 + u_1(x) \Big(2 x^2 - 6 x^2 + 4 x^2 \Big) + u_2'(x) \Big( 2 x^3 + x^3\ln(x)  \Big)  \\
	+ u_2(x) \Big( 2 x^2 \ln(x) + 2 x^2 + x^2 - 6 x^2 \ln(x) - 3 x^2  + 4 x^2 \ln(x) \Big)    &= \ln(x) \\[1.5ex]
	2 u_1'(x) x^3 + u_2'(x) \Big( 2 x^3 + x^3\ln(x)  \Big)   &= \ln(x)
\end{aligned}
$$

We can rearrange this equation to get:

$$
\begin{aligned}
2 u_1'(x) x^3 + u_2'(x) \Big( 2 x^3 + x^3\ln(x)  \Big) &= \ln(x) \\
u_1'(x) x^3 +  2 u_2'(x)  x^3 + x \underbrace{\Big(u_1'(x) x^2 +  u_2'(x)  x^2\ln(x)\Big)}_{=0} &= \ln(x)
\end{aligned}
$$

where the last term on the right hand side contains the auxiliary condition and so vanishes. As a result, this simplifies to:

$$ u_1'(x) +  2 u_2'(x) = x^{-3} \ln(x) $$

Now we have the system:

$$
\begin{aligned}
	u_1'(x) + u_2'(x) \ln(x) &= 0 \\
	u_1'(x) +  2 u_2'(x) &= x^{-3} \ln(x)
\end{aligned}
$$


Subtracting these two equations gives:

$$
u_2'(x) \Big(2 - \ln(x) \Big)  = x^{-3} \ln(x) \quad\implies\quad u_2'(x)  =  \frac{x^{-3} \ln(x)}{2 - \ln(x)}
$$

and from the first equation in the system of equations:

$$
u_1'(x) + u_2'(x) \ln(x) = 0 \quad\implies\quad u_1'(x) = - u_2'(x) \ln(x) \quad\implies\quad u_1'(x) = - \frac{x^{-3} \ln(x)^2}{2 - \ln(x)}
$$


These can be integrated and put back into the particular solution. With these we essentially have the solution to this ODE 







##  Practical Considerations and Limitations

The method of variation of parameters is a powerful tool for solving inhomogeneous second-order ODEs. However, as we have seen, it is not very efficient and sometime there are more practical choices avaluable to get the particular solution. In this section, we discuss when it is useful, when it becomes cumbersome, and how it compares to other methods.

###  When is Variation of Parameters Preferable?

Variation of parameters is particularly useful in the following scenarios:

#### Non-Polynomial, Non-Exponential, Non-Trigonometric Forcing Functions:

The method of undetermined coefficients is limited to forcing terms of the form:

$$
f(x) = P_n(x), \quad f(x) = e^{ax}, \quad f(x) = \cos(bx), \quad f(x) = \sin(bx),
$$

or products of these. If the forcing function $ f(x) $ is something more complicated, such as a logarithmic function, an arbitrary power of $ x $, or a special function (e.g., Bessel functions, Airy functions, etc.), undetermined coefficients is not really feasible. In these cases, variation of parameters is often the best option.
	
#### Forcing Functions that Resemble the Homogeneous Solution:

If the forcing function $ f(x) $ is structurally similar to the homogeneous solutions, the method of undetermined coefficients requires modifications (multiplying by $ x $, sometimes even by $ x^2 $). With variation of parameters, no such adjustments are needed, making it a more straightforward choice.
	
#### Generalizing to Higher-Order or Variable Coefficient Equations:

The variation of parameters method is not limited to second-order constant-coefficient ODEs. It extends to:

- **Higher-order linear ODEs**, where one seeks solutions of the form:
	
	$$
	y_p(x) = u_1(x) y_1(x) + u_2(x) y_2(x) + \dots + u_n(x) y_n(x).
	$$

- **Variable-coefficient ODEs**, where the method remains applicable because it does not rely on simple *ansatz* guesses.




###  When Does Variation of Parameters Become Cumbersome?

Despite its generality, the method of variation of parameters can be cumbersome in some cases:

#### Integral Complexity:

The primary challenge of variation of parameters is evaluating the integrals involved in:

$$
u_1(x) = -\int \frac{f(x) y_2(x)}{W(y_1, y_2)(x)} \, dx \qquad
u_2(x) = \int \frac{f(x) y_1(x)}{W(y_1, y_2)(x)} \, dx
$$

If the integrals are difficult (e.g., requiring multiple integration by parts or substitution), this method can become impractical. In such cases, alternative numerical methods or approximate solutions may be more feasible.
	
#### Computational Cost Compared to Undetermined Coefficients:

If the forcing function $ f(x) $ is well-suited to undetermined coefficients (i.e., it fits the table of standard forms), then undetermined coefficients is usually the faster method because it avoids integration entirely. The method of variation of parameters requires setting up and solving multiple integrals, which can add to computational effort.
	
#### Interpretability and Physical Insight:

The method of variation of parameters often leads to solutions with complex-looking integrals, making it harder to extract intuitive physical insights from the solution. In contrast, the method of undetermined coefficients often yields explicit polynomial, exponential, or trigonometric solutions that are easier to interpret in applied settings.


###  Comparison: Variation of Parameters vs. Undetermined Coefficients

| Method | Strengths | Weaknesses |
|---|---|---|
| Undetermined Coefficients | • Fast and straightforward when $f(x)$ is polynomial, exponential, or trigonometric. | • Cannot handle general functions such as logarithms or special functions.<br>• Requires modification when $f(x)$ overlaps with homogeneous solutions. |
| Variation of Parameters | • Works for any $f(x)$, including logarithms and special functions.<br>• More general and extends to variable-coefficient and higher-order ODEs. | • Involves computing integrals, which may be complicated or not expressible in closed form.<br>• More computationally intensive. |





















## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


Consider the second-order differential equation:

$$
y'' - 4y' + 4y = \frac{e^{2x}}{x}
$$

where $ x > 0 $. 


a) Find the general solution of the associated homogeneous ODE:

$$
y'' - 4y' + 4y = 0
$$
	
b) Using the homogeneous solutions $ y_1(x) $ and $ y_2(x) $, express the particular solution $ y_p(x) $ in the form:

$$
y_p(x) = u_1(x)y_1(x) + u_2(x)y_2(x)
$$

Derive the system of equations for $ u_1'(x) $ and $ u_2'(x) $ using the variation of parameters method.
	
c) Evaluate the Wronskian $ W(y_1, y_2) $.
	
d) Find explicit expressions for $ u_1(x) $ and $ u_2(x) $ by computing the necessary integrals.
	
e) Combine the homogeneous and particular solutions to express the full general solution.
















