---
layout: default
title: Mathematical Methods - Lecture 15
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 15
---


# Lecture 15 – Homogeneous and Inhomogeneous Second-Order ODEs - Undetermined Coefficients


## Additional Advanced Techniques for Homogeneous Second-Order ODEs

We have already explored how damping affects the behavior of homogeneous second-order linear ODEs and how to classify systems as underdamped, critically damped, or overdamped. One additional technique that can be very useful, especially when one solution of the homogeneous equation is known, is called the **reduction of order** method. In addition to this, we will also examine the Wronskian, which is a tool used to test if two, or more, functions are linearly independent. 

### Reduction of Order

Reduction of order allows you to find a another, linearly independent solution to the ODE without having to completely solve the full characteristic equation or, in the worst case, the full ODE via brute force. This method assumes you know one solution to the ODE already. Whether you found this solution by solving part of the characteristic equation or by making a good guess, it doesn't matter. All we care about is that we have one known solution to the differential equation we are trying to solve. 

Using the solution we already know $y_1(x)$ we can guess another solution $ y_2(x) $ of the form:

$$
y_2(x) = v(x) \, y_1(x)
$$

where $ v(x) $ is an unknown function that we will need to find. By substituting this form into the original ODE and using the fact that $ y_1(x) $ is a solution, we can derive an ODE for $ v(x) $ that is typically of a lower order than the initial ODE was and can hopefully be solved using simpler techniques. **This process works for variable-coefficient ODEs** as well, making this method a very valuable and general process to have in your tool kit. Let's see how it plays out in a general case.

Suppose you have the following homogeneous, constant-coefficient, linear ODE:

$$
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = 0
$$

and that you know one of the solutions to be $ y_1(x) $, meaning:

$$
a\frac{d^2y_1}{dx^2} + b\frac{dy_1}{dx} + c\,y_1 = 0
$$

First we need to find the first and second derivatives of the proposed second solution:

$$
\begin{aligned}
	y_2(x) &= v(x) \, y_1(x) \\[1.25 ex]
	\frac{dy_2}{dx} &= \frac{dv}{dx} \, y_1 +  v \, \frac{dy_1}{dx} \\[1.25 ex]
	\frac{d^2y_2}{dx^2} &= \frac{d^2v}{dx^2} \, y_1 + 2 \frac{dv}{dx} \, \frac{dy_1}{dx}  + v \, \frac{d^2y_1}{dx^2}
\end{aligned}
$$

Put this into the ODE:

$$
\begin{aligned}
	a\frac{d^2y_2}{dx^2} + b\frac{dy_2}{dx} + c\,y_2 &= 0 \\[1.25 ex]
	a\left(\frac{d^2v}{dx^2} \, y_1 + 2 \frac{dv}{dx} \, \frac{dy_1}{dx}  + v \, \frac{d^2y_1}{dx^2}\right) + b\left(\frac{dv}{dx} \, y_1 +  v \, \frac{dy_1}{dx}\right) + c \, v \, y_1 &= 0 
\end{aligned}
$$


and rearrange:

$$
\begin{aligned}
	a\frac{d^2v}{dx^2} \, y_1 + 2 a\frac{dv}{dx} \, \frac{dy_1}{dx}  + a v \, \frac{d^2y_1}{dx^2} + b\frac{dv}{dx} \, y_1 +  b v \, \frac{dy_1}{dx} + c \, v \, y_1 &= 0\\[1.25ex]
	a \ y_1 \ \frac{d^2v}{dx^2} + \left(2 a \, \frac{dy_1}{dx} + b \, y_1\right) \ \frac{dv}{dx}  +  \underbrace{\left( a \, \frac{d^2y_1}{dx^2} +  b \, \frac{dy_1}{dx} + c \, y_1 \right)}_{=0} \ v  &= 0 
\end{aligned}
$$

where the last term is zero because $y_1(x)$ is a solution to the homogeneous ODE. Simplifying gives:

$$ a\frac{d^2v}{dx^2} \, y_1 + \left(2 a \, \frac{dy_1}{dx} + b \, y_1\right) \frac{dv}{dx} = 0 $$

This can be turn into a first-order ODE using the following substitution:

$$ u(x) = \frac{dv}{dx} \qquad \implies \qquad  a\frac{du}{dx} \, y_1 + \left(2 a \, \frac{dy_1}{dx} + b \, y_1\right) u = 0  $$

which we can solve using our standard first-order solutions techniques, like an integrating factor, for example. 



{% capture ex %}
Consider the homogeneous ODE

$$
y'' - y' - 2y = 0
$$

Suppose we already know one solution is

$$
y_1(x) = e^{2x}
$$

Our goal is to find a second, linearly independent solution $ y_2(x) $ using the reduction of order method.

Using the proposed second solution:

$$
y_2(x) = v(x) \, y_1(x) = v(x) \, e^{2x}
$$

where $ v(x) $ is an unknown function. Computing the derivatives:

$$
\begin{aligned}
	y_2'(x) &= v'(x)e^{2x} + 2v(x)e^{2x} \\[1.25ex]
	y_2''(x) &= v''(x)e^{2x} + 4v'(x)e^{2x} + 4v(x)e^{2x}
\end{aligned}
$$

Substituting these into the original ODE and simplifying as much as possible:

$$
\begin{aligned}
	y'' - y' - 2y &= 0 \\[1.25ex]
	\Bigl( v''(x)e^{2x} + 4v'(x)e^{2x} + 4v(x)e^{2x} \Bigr) - \Bigl( v'(x)e^{2x} + 2v(x)e^{2x} \Bigr) - 2\Bigl( v(x)e^{2x} \Bigr) &= 0 \phantom{\hspace{5cm}} \\[1.25ex]
	v''(x)e^{2x} + 4v'(x)e^{2x} + 4v(x)e^{2x} - v'(x)e^{2x} - 2v(x)e^{2x} - 2v(x)e^{2x} &= 0  \\[1.25ex]
	v''(x)e^{2x} + 3v'(x)e^{2x} &= 0  \\[1.25ex]
	\Bigl(v''(x) + 3v'(x)\Bigr)e^{2x} &= 0  \\[1.25ex]
	v''(x) + 3v'(x) &=0
\end{aligned}
$$

where we have forced $v''(x) + 3v'(x)$ to be zero since asking $e^{2x}$ to be zero would only be true at $ \rightarrow -\infty$.

Notice the function $v(x)$ itself does not show up in this new relation. we can take advantage of this by making the following substitutions $ w(x) = v'(x) $, so that $ w'(x) = v''(x) $. The equation becomes:

$$
w'(x) + 3w(x) = 0
$$

This is a first-order linear ODE for $ w(x) $. Using separation of variables:

$$
\frac{dw}{dx} = -3w \quad \Longrightarrow \quad \frac{dw}{w} = -3\,dx
$$
Integrate both sides:

$$
\ln|w| = -3x + C_1 \quad \Longrightarrow \quad w(x) = Ae^{-3x}
$$
where $ A = e^{C_1} $ is an arbitrary constant.

Since $ w(x) = v'(x) = Ae^{-3x} $, integrate to find $ v(x) $:

$$
v(x) = \int Ae^{-3x}\,dx = -\frac{A}{3}e^{-3x} + B
$$

where $ B $ is another constant of integration. Thus, we have found the second solution t the differential equation to be:

$$
y_2(x) = v(x)e^{2x} = \left(-\frac{A}{3}e^{-3x} + B\right)e^{2x} = -\frac{A}{3}e^{-x} + B e^{2x}
$$

Since $ B e^{2x} $ is just a multiple of the known solution $ y_1(x) = e^{2x} $, we discard that term (or absorb it into the constant for the homogeneous solution). We will also ignore the coefficient of the first term since it can all be absorbed into the unknown constant in the general solution. This gives:

$$
y_2(x) = e^{-x}
$$

**Conclusion:** 

Using reduction of order, we have found a second, linearly independent solution:

$$
y_2(x) = e^{-x}
$$

so the general solution to the ODE is:

$$
y(x) = C_1 y_1(x) + C_2 y_2(x)
$$

$$
y(x) = C_1e^{2x} + C_2e^{-x}
$$

{% endcapture %}
{% include example.html content=ex %}




#### Reduction of Order vs. Variation of Parameters

You may recall the method we covered previously called variation of parameters, which looks very similar to this process. In fact, reduction of order is can be thought of as a subset of the more general variation of parameters method. Both are powerful techniques in the study of differential equations, but they are used in distinctly different contexts and serve different purposes, which is why we give them different names.

| Feature | Reduction of Order | Variation of Parameters |
|---|---|---|
| **Context** | Used for homogeneous equations. | Primarily used for inhomogeneous equations. |
| **Purpose** | Finds a second linearly independent solution $y_2(x)$ when one solution $y_1(x)$ of a homogeneous second-order linear ODE is already known. The general solution would then be: $$ y(x) = C_1y_1(x) + C_2y_2(x) $$ | Finds a particular solution $y_p(x)$ to an inhomogeneous second-order linear ODE: $$ a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = f(x) $$ using the homogeneous solutions $y_1(x)$ and $y_2(x)$. | 
| **Assumption** | Assume the second solution has the form: $$ y_2(x) = v(x)\,y_1(x) $$ | Assume a particular solution of the form: $$ y_p(x) = u_1(x)y_1(x) + u_2(x)y_2(x) $$ |
| **Key Idea** | The unknown function $v(x)$ is determined by substituting into the ODE, which reduces the problem to a first-order equation for $v(x)$. | The functions $u_1(x)$ and $u_2(x)$ are determined by substituting into the ODE and imposing the auxiliary condition: $$ u_1'(x)y_1(x) + u_2'(x)y_2(x) = 0 $$ |



While both methods leverage known solutions to help build the full solution, reduction of order is tailored specifically for finding an additional homogeneous solution when one is already known, whereas variation of parameters is aimed at finding a particular solution for inhomogeneous ODEs. 








### Wronskian Analysis

It was mentioned in the previous section that variation of parameters required two functions to be linearly independent. What does that mean, and how can we check for it? 

Two functions, similar to two vectors, are said to be linearly independent if the following:

$$ a y_1(x) + b y_2(x) = 0 $$

is only possible if the constants $a$ and $b$ both equal zero. That is, no combination of values for $a$ and $b$ will make this equation true other than $a=0$ and $b=0$. There is a small issue here. This condition for linear independence is a condition on the constants $a$ and $b$. What we would really like is a condition on the functions we can use to test for linear independence. 

We can find such a condition on the functions by forcing new equations from the linearly independent check above. Notice we can generate another equation involving the functions by taking the derivative of the previous equation: 

$$ a y_1'(x) + b y_2'(x) = 0 $$

This leave us with the set of equations:

$$
\begin{aligned}
	a y_1(x) + b y_2(x) &= 0 \\
	a y_1'(x) + b y_2'(x) &= 0
\end{aligned}
$$

which can be written in matrix form as:

$$ \begin{bmatrix}
	y_1(x)  & y_2(x) \\ y_1'(x) & y_2'(x)
\end{bmatrix} \begin{bmatrix}
a \\ b
\end{bmatrix} = \begin{bmatrix}
0 \\ 0 
\end{bmatrix} $$

We can solve for $a$ and $b$ by taking the inverse of the matrix on the left hand side of the equation, and applying it to on the left of both sides of the equation to get:

$$
\begin{aligned}
	\begin{bmatrix}
		y_1(x)  & y_2(x) \\ y_1'(x) & y_2'(x)
	\end{bmatrix} \begin{bmatrix}
		a \\ b
	\end{bmatrix} &= \begin{bmatrix}
		0 \\ 0 
	\end{bmatrix}  \\[1.5ex]
	\begin{bmatrix}
		y_1(x)  & y_2(x) \\ y_1'(x) & y_2'(x)
	\end{bmatrix}^{-1} \begin{bmatrix}
		y_1(x)  & y_2(x) \\ y_1'(x) & y_2'(x)
	\end{bmatrix} \begin{bmatrix}
		a \\ b
	\end{bmatrix} &= \begin{bmatrix}
		y_1(x)  & y_2(x) \\ y_1'(x) & y_2'(x)
	\end{bmatrix}^{-1} \begin{bmatrix}
		0 \\ 0 
	\end{bmatrix}  \\[1.5ex]
	\begin{bmatrix}
		a \\ b
	\end{bmatrix} &= \begin{bmatrix}
		0 \\ 0 
	\end{bmatrix} 
\end{aligned}
$$

By assuming the matrix on the left had an inverse, we were able to get the solution $a=0$ and $b = 0$, which is the condition that implies the two functions are linearly independent. This is the condition on the functions we were looking for! Recall, for this matrix to have an inverse, its determinant must be nonzero! Thus, $y_1(x)$ and $y_2(x)$ are linearly independent if:

$$ \begin{vmatrix}
	y_1(x)  & y_2(x) \\ y_1'(x) & y_2'(x)
\end{vmatrix} \ne 0 $$

{% capture ex %}
To check if two functions are linearly independent, construct the following matrix for the functions $y_1(x)$ and $y_2(x)$:

$$ \begin{bmatrix}
	y_1(x)  & y_2(x) \\ y_1'(x) & y_2'(x)
\end{bmatrix} $$

**These functions will be linearly independent if:**

$$ \begin{vmatrix}
	y_1(x)  & y_2(x) \\ y_1'(x) & y_2'(x)
\end{vmatrix} \ne 0 $$

{% endcapture %}
{% include result.html content=ex %}


This process can be generalized into one of the most powerful tools in the analysis of ODEs, the **Wronskian**. The Wronskian allows us to verify the linear independence of two solutions, which is crucial since the general solution of a second-order homogeneous ODE should be a linear combination of two linearly independent functions.

**Definition:** The Wronskian for the two functions $ y_1(x) $ and $ y_2(x) $ is defined as:

$$
W(y_1, y_2)(x) = \begin{vmatrix}
	y_1(x) & y_2(x) \\
	y_1'(x) & y_2'(x)
\end{vmatrix} = y_1(x)y_2'(x) - y_2(x)y_1'(x)
$$

If $ W(y_1, y_2)(x) \neq 0 $ for all values $ x $ in the interval of interest, then $ y_1 $ and $ y_2 $ are linearly independent.

What do we mean by, "... in the interval of interest"? By this we mean, that the Wronskian can be zero. But, if that happens for values of $x$ that we do not care about for our model, then we can ignore it and work as if the functions are linearly independent. It turns out this is just how linear independence for functions works: they can be independent for some values of $x$, but do not have to be for all values of $x$. As long as the functions are independent for the interval of $x$ values we are interested in, then we are fine.

This process can be generalized to any number of functions by simply taking more derivatives and adding more rows. Since you need the same number of rows as columns to take the determinant, if you have $n$ functions you are testing all together, then you will need to calculate up to the $n$-th derivative for each function. In this can the Wronskian will be:

$$
W(y_1, y_2, \dots, y_n)(x) = \begin{vmatrix}
	y_1(x) & y_2(x) & \cdots & y_n(x) \\
	y_1'(x) & y_2'(x) & \cdots & y_n'(x) \\
	\vdots &\vdots &  & \vdots &\\
	y_1^{(n)}(x) & y_2^{(n)}(x) & \cdots & y_n^{(n)}(x) \\
\end{vmatrix}
$$

If $W(y_1, y_2, \dots, y_n)(x) \ne 0$ for all $ x $ in the interval of interest, then $ y_1 $, $ y_2 $,... $ y_n $ are all linearly independent.

The Wronskian provides a quick diagnostic to confirm that the solutions obtained are not merely multiples of one another. This is particularly useful when employing methods like reduction of order, where one solution is known and the second solution must be verified for independence. Without linear independence, our general solution would fail to capture the full behavior of the system.

{% capture ex %}
Consider the undamped mass-spring system governed by:

$$
\frac{d^2y}{dx^2} + \omega^2 y = 0
$$

You can show that the following functions are solutions to this ODE: 

$$
y_1(x) = \cos(\omega x) \quad \text{and} \quad y_2(x) = \sin(\omega x)
$$

Let's check to see if they are linearly independent by taking their derivatives:

$$
y_1'(x) = -\omega \sin(\omega x) \quad \text{and} \quad y_2'(x) = \omega \cos(\omega x)
$$

and calculating the Wronskian:

$$
\begin{aligned}
	W(y_1, y_2)(x) &= y_1(x)y_2'(x) - y_2(x)y_1'(x) \\ 
	&= \cos(\omega x)\Bigl(\omega \cos(\omega x)\Bigr) - \sin(\omega x)\Bigl(-\omega \sin(\omega x)\Bigr) \\
	&= \omega \left[\cos^2(\omega x) + \sin^2(\omega x)\right] \\
	&= \omega
\end{aligned}
$$

Since $ \omega \neq 0 $, the Wronskian is nonzero for all $ x $, confirming that $ y_1 $ and $ y_2 $ are linearly independent.
{% endcapture %}
{% include example.html content=ex %}







## Introduction to Inhomogeneous Second-Order ODEs

In many physical systems, external forces or inputs modify the natural behavior described by homogeneous second-order ODEs. These modifications lead to **inhomogeneous** or **forced** second-order ODEs, which take the general form:

$$
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = f(x)
$$

where $ f(x) $ represents the external driving force or input. This driving term is responsible for the **steady-state** (or forced) response of the system, while the homogeneous part of the solution still represents the **transient** behavior.

As we saw with inhomogeneous first-order ODEs, instead of trying to solve this problem all at once, we generally break it into pieces. Specifically, find a homogeneous solution $y_h(x)$ to take care of the transient behavior: 

$$
a\frac{d^2y_h}{dx^2} + b\frac{dy_h}{dx} + c\,y_h = 0
$$

The solution $y_h(x)$ will have two unknown constants that will be used to incorporate the initial conditions of the problem. 

This solution represented the **transient response** $ of the system. That is, this is the system's natural behavior and the response is determined by the initial conditions. In a damped system, this part of the general solution typically decays over time and eventually vanishes.

The forced behavior will be covered by the particular solution $y_p(x)$ whose sole purpose is the satisfy the inhomogeneous ODE:

$$
a\frac{d^2y_p}{dx^2} + b\frac{dy_p}{dx} + c\,y_p = f(x)
$$

This solution represent the **steady-state response** of the system to the driving terms and will not have any unknown constants since it is fully defined by the driving term. For example, if $ f(x) $ is sinusoidal, $ y_p(x) $ will usually be a sinusoidal function at the same frequency as $ f(x) $, but with a phase shift and possibly a different amplitude. 

A particularly interesting phenomenon occurs when the driving term is sinusoidal and has a frequency equal to the natural frequency of the system. In this case, known as **resonance**, the amplitude of the steady-state response can increase dramatically. In practical applications, resonance must be carefully managed, as excessive oscillations can lead to system failure of the parts which make up the physical system.

Since this is a linear ODE, the general solution can be found being simply adding these two solutions together to get:

$$
y(x) = y_h(x) + y_p(x)
$$











### Methods for Finding Particular Solutions

The homogeneous part $ y_h(x) $ captures the natural response of the system, and we have already covered techniques for solving this problem. Finding the particular solution $ y_p(x) $ is key to understanding how external forces drive the system. Two widely used methods to obtain $ y_p(x) $ are the **method of undetermined coefficients** and the **method of variation of parameters**.

#### Method of Undetermined Coefficients

This method works well when the driving function $ f(x) $ in the inhomogeneous ODE

$$
a\frac{d^2y}{dx^2} + b\frac{dy}{dx} + c\,y = f(x)
$$

is of a simple, standard form (e.g., polynomials, exponentials, sines, or cosines). The idea is to **guess** a form for $ y_p(x) $ that mimics the structure of $ f(x) $, but with undetermined coefficients. This guess is substituted into the ODE, and you can solve for these coefficients by matching terms. This is sometimes referred to as the "guess and check" method. The critical point here is that **the undetermined coefficients are constants**, and do not depend on the working variable. If you make a guess, substitute it into the ODE, and find that the undetermined coefficients depend on the working variable, then you have to restart the problem by making another guess. 

When solving a inhomogeneous second-order ODE using the method of undetermined coefficients, the form of the driving term $ f(x) $ guides our guess for the particular solution $ y_p(x) $. This takes practice to get comfortable with, but the guesses are normally well informed. The table below summarizes several common types of $ f(x) $ and the corresponding *ansatz*. *Ansatz* is a fancy word for ``guess'' you sometimes see thrown around when people want to avoid saying they guessed the answer. Here is a nice table of *ansatz* for $ y_p(x) $ given some specific forms of possible driving terms.


| Driving Term $f(x)$ | Guess for Particular Solution $y_p(x)$ |
|---|---|
| $P_n(x) = 2x^n + 3x^{n-1} + \cdots - 9x + 2$ | $Q_n(x) = a_nx^n + a_{n-1}x^{n-1} + \cdots + a_1x + a_0$ |
| $e^{ax}$ | $Ae^{ax}$ |
| $\cos(bx)$ or $\sin(bx)$ | $A\cos(bx) + B\sin(bx)$ |
| $e^{ax}\cos(bx)$ or $e^{ax}\sin(bx)$ | $e^{ax}\left(A\cos(bx) + B\sin(bx)\right)$ |
| $P_n(x)e^{ax}$ | $e^{ax}Q_n(x)$ |
| $P_n(x)\cos(bx)$ or $P_n(x)\sin(bx)$ | $Q_n(x)\cos(bx) + R_n(x)\sin(bx)$ |

**Note:** $P_n(x)$, $Q_n(x)$, and $R_n(x)$ represent polynomials of degree $n$.

**Important Note:** If your initial guess for $y_p(x)$ turns out to already be a solution for the homogeneous equation, multiply your guess by $x$ (or higher powers of $x$ if necessary) to ensure linear independence.

This table is a handy reference to quickly determine the form of your particular solution based on the driving term, streamlining your approach to solving nonhomogeneous ODEs in physics and engineering.


{% capture ex %}
Consider the ODE

$$
y'' + 4y = \cos(2x)
$$

The homogeneous problem for this ODE is:

$$
y'' + 4y = 0
$$

which has the solution (you should check this!):

$$
y_h(x) = A\cos(2x) + B\sin(2x)
$$

To find the particular solution we can use the method of undetermined coefficients and guess a solution. Checking our table reveals that a reasonable guess for the particular solution would be:

$$
y_p(x) = C\cos(2x) + D\sin(2x)
$$

but this is the same as the homogeneous solution. The Note under the *ansatz* table suggests simply multiplying this guess by $x$ and trying that instead. This gives: 

$$
y_p(x) = x \Big(C\cos(2x) + D\sin(2x)\Big) = C x \cos(2x) + D x \sin(2x)
$$ 

Before we substitute $ y_p(x) $ into the ODE, let's calculate its derivatives: 

$$
\begin{aligned}
	y_p'(x) &= C \cos(2x) + D \sin(2x) - 2 C x \sin(2x) + 2 D x \cos(2x) \\[1.25ex]
	y_p''(x) &= -4C \sin(2x) + 4 D \cos(2x) - 4 C x \cos(2x) - 4 D x \sin(2x)
\end{aligned} 
$$

Putting these into the ODE, collect like terms:

$$
\begin{aligned}
	y_p'' + 4y_p &= \cos(2x) \\[1.5ex]
	-4C \sin(2x) + 4 D \cos(2x) - 4 C x \cos(2x) - 4 D x \sin(2x) + 4\Big(C x \cos(2x) + D x \sin(2x) \Big) &= \cos(2x) \\[1.5ex]
		-4C \sin(2x) + 4 D \cos(2x) - 4 C x \cos(2x) - 4 D x \sin(2x) + 4 C x \cos(2x) + 4 D x \sin(2x) &= \cos(2x) \\[1.5ex]
		-4C \sin(2x) + 4 D \cos(2x) &= \cos(2x) 
\end{aligned}
$$

From here we can see that $C = 0$ and $ D = \tfrac{1}{4}$ offers a simple solution for our undetermined coefficients. 

Thus, our particular solution to this problem will be:

$$
y_p(x) = \frac{1}{4} \,  x \,  \sin(2x)
$$ 

and the general solution will be:

$$
\begin{aligned}
	y(x) &= y_h(x) + y_p(x) \\
	y(x) &= A\cos(2x) + B\sin(2x) + \frac{1}{4} \,  x \,  \sin(2x)
\end{aligned}
$$

At this point we would apply any initial conditions given for the problem. 
{% endcapture %}
{% include example.html content=ex %}













### Specific Example: Forced Harmonic Oscillator

In many real-world systems, an external force continually acts on a system, which drives it away from its natural response. The classic example is the forced harmonic oscillator, which models situations such as a mass-spring system subject to an external periodic force, or an RLC circuit driven by an AC voltage source.

A forced harmonic oscillator is typically governed by an inhomogeneous, second-order ODE of the form:

$$
m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F(t)
$$

where:

- $ m $ is the mass (or inductance in an electrical circuit),
- $ c $ is the damping coefficient (or resistance),
- $ k $ is the spring constant (or 1 over the capacitance), and
- $ F(t) $ is the external driving force (or applied voltage).

Let's work through an example with numbers to keep any potential mathematical vagueness to a minimum.

{% capture ex %}
Consider a mass-spring system with $ m = 1\,\text{kg} $, $ c = 0.5\,\text{N}\cdot\text{s/m} $, and $ k = 4\,\text{N/m} $ subject to a driving force:

$$
F(t) = 2 \cos(1.5t)
$$

The governing equation is:

$$
\frac{d^2x}{dt^2} + 0.5\frac{dx}{dt} + 4x = 2\cos(1.5t)
$$

First, solve the homogeneous equation:

$$
\frac{d^2x}{dt^2} + 0.5\frac{dx}{dt} + 4x = 0
$$

The characteristic equation is:

$$
r^2 + 0.5r + 4 = 0
$$

and applying the quadratic equation:

$$
\begin{aligned}
r &= \frac{-0.5 \pm \sqrt{(0.5)^2 - 4 (1) (4)}}{2(1)} \\
&= \frac{-0.5 \pm \sqrt{0.25 - 16}}{2} \\
&= -0.25 \pm \frac{1}{2} \sqrt{0.25 - 16} \\
&= -0.25 \pm \frac{1}{2} \sqrt{-15.75} \\
&= -0.25 \pm i \ \  \frac{1}{2} \sqrt{15.75}
\end{aligned}
$$

We find the complex conjugate roots $ r = -0.25 \pm i\,\omega_d $ (with $ \omega_d = \frac{1}{2} \sqrt{15.75} $). This leads to a homogeneous solution of the form:

$$
x_h(t) = e^{-0.25t} \Big( C_1 \cos(\omega_d t) + C_2 \sin(\omega_d t) \Big)
$$

Next, find a particular solution $ x_p(t) $. Since the forcing term is $ 2\cos(1.5t) $, a reasonable guess would be:

$$
x_p(t) = A\cos(1.5t) + B\sin(1.5t)
$$

Neither of these are homogeneous solutions, so we are free to use this guess as is. 

Now we can substitute $ x_p(t) $ into the original ODE, 

$$ x_p'(t) = -1.5A\sin(1.5t) + 1.5 B\cos(1.5t)  $$

$$ x_p''(t) = -(1.5)^2A\cos(1.5t) - (1.5)^2 B\sin(1.5t)  $$

gives:

$$
\begin{aligned}
	\frac{d^2x}{dt^2} + 0.5\frac{dx}{dt} + 4x &= 2\cos(1.5t) \\[1.25ex]
	\Big(-(1.5)^2A\cos(1.5t) - (1.5)^2 B\sin(1.5t)\Big) + 0.5\Big(-1.5A\sin(1.5t) + 1.5 B\cos(1.5t)\Big) \\
	\hspace{2cm} + 4\Big(A\cos(1.5t) + B\sin(1.5t)\Big) &= 2\cos(1.5t) \\[1.25ex]
	\Big(-(1.5)^2 A + 0.75 B + 4A\Big)\cos(1.5t) + \Big(-(1.5)^2 B - 0.75 A + 4B\Big)\sin(1.5t) &= 2\cos(1.5t) \\[1.25ex]
	\Big(1.75 A + 0.75 B\Big)\cos(1.5t) + \Big(1.75 B - 0.75 A\Big)\sin(1.5t) &= 2\cos(1.5t)
\end{aligned}
$$

This gives the conditions:

$$ 1.75 A + 0.75 B = 2 \qquad\text{and}\qquad 1.75 B - 0.75 A = 0  $$

which we can multiply through by 4 to get:

$$ 7 A + 3 B = 8 \qquad\text{and}\qquad 7 B - 3 A = 0  $$

We can solve the second equation for $A$ to get:

$$ 7 B - 3 A = 0 \quad\implies\quad A = \frac{7}{3} \,  B $$

and put that into the first equation to get $B$: 


$$ 7 \Big(\frac{7}{3} \,  B\Big) + 3 B = 8 \quad\implies\quad  49 B + 9 B = 24 \quad\implies\quad B = \frac{12}{29}$$

which give $A$ as:

$$ A = \frac{7}{3} \,  \left(\frac{12}{29}\right) = \frac{28}{29} $$

This leaves us with a particular solution of:

$$ x_p(t) = \frac{28}{29} \, \cos(1.5t) + \frac{12}{29} \, \sin(1.5t) $$

Our general solution for this problem is thus:

$$ x(t) = x_h(t) + x_p(t) = e^{-0.25t} \Big( C_1 \cos(\omega_d t) + C_2 \sin(\omega_d t) \Big) + \frac{28}{29} \, \cos(1.5t) + \frac{12}{29} \, \sin(1.5t) $$

{% endcapture %}
{% include example.html content=ex %}







## More Worked Examples

Let's look over a couple more specific examples of these techniques to help hammer them home.

### Example 1: Simple Exponential

Consider the differential equation:

$$
y'' - y' - 2y = 2e^{3x}
$$

First, as is always the first step when solving inhomogeneous ODEs, we solve the associated homogeneous equation is:

$$
y'' - y' - 2y = 0
$$

The characteristic equation is:

$$
r^2 - r - 2 = 0
$$

which factors as:

$$
(r - 2)(r + 1) = 0 \qquad\implies\qquad r = 2 \quad\text{and}\quad r = -1
$$

Thus, the homogeneous solution is:

$$
y_h(x) = C_1 e^{2x} + C_2 e^{-x}
$$


For the particular solution we will use the method of undetermined coefficients. The forcing term is $2e^{3x}$. Since $3$ is not a root of the characteristic equation, we guess:

$$
y_p(x) = Ae^{3x}
$$

Compute the derivatives:

$$
y_p'(x) = 3Ae^{3x} \qquad y_p''(x) = 9Ae^{3x}
$$

Substitute into the ODE:

$$
9Ae^{3x} - 3Ae^{3x} - 2Ae^{3x} = (9 - 3 - 2)Ae^{3x} = 4Ae^{3x}.
$$

Set equal to the right-hand side:

$$
4Ae^{3x} = 2e^{3x}
$$

Thus, $4A = 2$ or $A = \frac{1}{2}$. This means our particular solution will be:

$$ y_p(x) = \frac{1}{2} e^{3x} $$


The general solution is the sum of the homogeneous and particular solutions:

$$
y(x) = y_h(x) + y_p(x) = C_1 e^{2x} + C_2 e^{-x} + \frac{1}{2}e^{3x}
$$




### Example 2: Polynomial Forcing Function

Consider the differential equation:

$$
y'' - 4y' + 4y = x^2
$$

Let's solve the associated homogeneous equation to get the homogeneous solution. First we have:

$$
y'' - 4y' + 4y = 0
$$

for which we have the following characteristic equation:

$$
r^2 - 4r + 4 = 0
$$

which factors as:

$$
(r - 2)(r - 2) = 0 \qquad \implies \qquad r = 2, \quad 2
$$

Since we have a **repeated root**, the general homogeneous solution takes the form:

$$
y_h(x) = C_1 e^{2x} + C_2 x e^{2x}
$$

Now for the particular solution. The forcing term is a polynomial: $ f(x) = x^2 $. According to our table of standard guesses, we would normally assume:

$$
y_p(x) = Ax^2 + Bx + C
$$

However, we must check whether any part of this guess overlaps with the homogeneous solution. Since neither $ x^2 $, $ x $, nor a constant appear in $ y_h(x) $, we can proceed with this guess.

Taking the first and second derivative:

$$
\begin{aligned}
	y_p'(x) &= 2Ax + B \\
	y_p''(x) &= 2A
\end{aligned}
$$

and substituting these into the inhomogeneous ODE:

$$
y'' - 4y' + 4y = x^2
$$

gives:

$$
\begin{aligned}
	(2A) - 4(2Ax + B) + 4(Ax^2 + Bx + C) &= x^2 \\[1.25ex]
	2A - 8Ax - 4B + 4Ax^2 + 4Bx + 4C &= x^2
\end{aligned}
$$

Rearranging by powers of $ x $:

$$
4Ax^2 + (-8A + 4B)x + (2A - 4B + 4C) = x^2
$$

By comparing coefficients with the right-hand side $ x^2 + 0x + 0 $:


- $ 4A = 1 $ (from $ x^2 $ terms) $ \Rightarrow A = \frac{1}{4} $
- $ -8A + 4B = 0 $  (from $ x $ terms) 

	$$ \Rightarrow -8\left(\frac{1}{4}\right) + 4B = 0 \Rightarrow -2 + 4B = 0  \Rightarrow B = \frac{1}{2} $$

- $ 2A - 4B + 4C = 0 $  (from constant terms) 

	$$ \Rightarrow 2\left(\frac{1}{4}\right) - 4\left(\frac{1}{2}\right) + 4C = 0 \Rightarrow -\frac{3}{2} + 4C = 0 \Rightarrow  C = \frac{3}{8} $$


which combine to get the particular solution:

$$
y_p(x) = \frac{1}{4} x^2 + \frac{1}{2} x + \frac{3}{8}
$$

The general solution to the original equation is:

$$
y(x) = C_1 e^{2x} + C_2 x e^{2x} + \frac{1}{4} x^2 + \frac{1}{2} x + \frac{3}{8}
$$











## Looking Ahead: When Guessing Doesn’t Work

We’ve seen that the **method of undetermined coefficients** is a powerful tool for solving inhomogeneous second-order ODEs, but it has limitations. This method relies on being able to **guess the form of the particular solution** based on the structure of $ f(x) $. However, there are cases where this approach fails or becomes impractical. 

For example, consider the ODE:

$$
y'' - y' - 2y = \ln(x)
$$

If we try to use our table to guess a form for $ y_p(x) $, we quickly realize that **there’s no simple template for logarithmic functions**. The same issue arises for functions like $ f(x) = \frac{1}{1+x^2} $ or $ f(x) = x e^x $.

To handle these more general cases, we need a **systematic approach that doesn’t rely on educated guessing**. This is where the **method of variation of parameters** comes in. Unlike undetermined coefficients, which assumes a particular solution of a fixed algebraic form, **variation of parameters finds the particular solution by allowing the coefficients themselves to vary dynamically**.

In the next lecture, we will:

- Derive the method of **variation of parameters** from first principles.
- Compare it directly to **undetermined coefficients** to clarify when each method is useful.
- Work through examples where variation of parameters is **the only viable method**.


By the end of the next lecture, we will have a complete toolbox for solving inhomogeneous second-order ODEs—allowing us to handle any forcing function we encounter.






## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.



Consider the second-order differential equation:

$$
y'' - 2y' + y = e^x + x^2
$$

a) Solve the associated **homogeneous** equation:

$$
y'' - 2y' + y = 0
$$

then use **reduction of order** to find a second, linearly independent solution $ y_2(x) $.


b) Compute the **Wronskian** of the solutions $ y_1(x) $ and $ y_2(x) $. Verify that the solutions are linearly independent.



c) Use the **method of undetermined coefficients** to find a particular solution $ y_p(x) $ for the inhomogeneous equation:

$$
y'' - 2y' + y = e^x + x^2
$$

*Hint:* Solve for the particular solution in two parts:
- Solve $ y'' - 2y' + y = e^x $
- Solve $ y'' - 2y' + y = x^2 $
- The final solution is the sum of the two.


d) Write the **general solution** $ y(x) $ to the given ODE by combining the homogeneous and particular solutions.


Additional Challenge) If the initial conditions are $ y(0) = 3 $ and $ y'(0) = -2 $, determine the values of the constants in the general solution.





