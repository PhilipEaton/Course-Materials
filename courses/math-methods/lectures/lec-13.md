---
layout: default
title: Mathematical Methods - Lecture 13
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 13
---



# Lecture 13 – Advanced Topics in First-Order ODEs


## Introduction

In our previous lectures, we discussed the basic methods for solving first-order ordinary differential equations—namely, separation of variables and the integrating factor method. These techniques served us well in tackling a variety of linear ODEs in contexts ranging from RC circuits to disease modeling. However, as we advance our study of differential equations, we will encounter cases and models that require more sophisticated approaches. 

Let's expand our analytical toolkit by exploring several advanced topics in first-order ODEs. Our goal is to handle a broader class of equations and gain a deeper conceptual understanding of their behavior in physical systems. The topics we will cover include:


- **Slope Field Diagrams:** Tools that help us understand the long-term behavior of ODEs without needing an explicit solution.
- **Bernoulli's Equation:** A special nonlinear ODE that, through a clever substitution, can be reduced to a linear ODE.
- **Variation of Parameters for First-Order ODEs:** An alternative method for solving inhomogeneous equations, complementing the integrating factor approach. We will see this method expanded upon when we go to solve second order ODEs.
- **Additional Real-World Applications:** A look at models from population dynamics, chemical kinetics, and economics that illustrate the diverse applicability of first-order ODEs.








## Slope Field Diagrams

Not all differential equations yield neat, closed-form solutions. In many cases, however, we do not need an explicit solution to understand the behavior of the system being modeled. Instead, qualitative analysis can provide valuable insights into the dynamics of a system. One particularly useful tool in qualitative analysis is the **slope field diagram**.

Qualitative analysis involves **using logical reasoning and a solid understanding of the physical context to predict the general behavior of solutions** without having to solve the equation completely. In other words, you use your intuition and deductions about the system to anticipate features such as equilibrium points, stability, and the overall direction of change.

For example, suppose you are solving an ODE that models the motion of a mass attached to a spring. From experience, you know the mass will oscillate when set into motion. This means, without having to solve the ODE explicitly, you can expect the solution to involve oscillatory functions like sine and cosine.

In addition to using logic to break down what general solutions should look like, we can use features such as equilibrium points, their stability, and the overall direction of change, to predict how a system evolves over time. This additional information can be nicely displayed in a **slope field diagram**.

Slope field diagrams are particularly useful tools for examining first-order ODEs without having to actually solve them. Consider a general first-order ODE:

$$
\frac{dy}{dt} = f(y,t)
$$

The equilibrium (or steady-state) solutions occur when: 

$$ \frac{dy}{dt} = 0 \implies f(y,t) = 0 $$

We can then solve this equation for $y$ as a function of $t$. In mathematics these lines are called the **nullclines**. Nullclines represent equilibrium points/lines for the ODE. The stability of these points/lines can be revealed using the following logic:

- If $ f(y,t) > 0 \implies \frac{dy}{dt} > 0 $ in the neighborhood of an equilibrium point, the solution will increase with time.
- If $ f(y,t) < 0 \implies \frac{dy}{dt} < 0 $ in the neighborhood, the solution will decrease in time.

We represent these behaviors on a slope field—a two-dimensional diagram where the equilibrium points/lines are plotted and each region is is labeled to indicate whether $ y $ is increasing or decreasing as time progresses.

### Constructing a Slope Field Diagram:


- Identify the nullclines by solving $ f(y) = 0 $.
- Draw the nullclines in a $y$ versus $t$ plot.
- Determine the sign of $ f(y) $ in each interval between these points.
- Use arrows to indicate whether solutions are increasing or decreasing in each region.



{% capture ex %}
Consider the ODE:

$$
\frac{dy}{dt} = y(1-y)
$$

Notice, for this ODE we have $f(y,t) = y(1-y)$. The equilibrium points/lines can be found in the following manner:
    
$$
y(1-y) = 0 \qquad \implies \qquad y = 0 \quad \text{ and }\quad  y = 1
$$ 

Drawing these nulclines gives:

<img
src="{{ '/courses/math-methods/images/lec13/SP_Example_01_1.png' | relative_url }}"
alt="Horizontal lines at y = 0 and y = 1."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">


Now, test the sign of $ f(y) = y(1-y) $:

- For $ y < 0 $, $ f(y) < 0 $ (solutions decrease).
- For $ 0 < y < 1 $, both $ y > 0 $ and $ 1-y > 0 $ so $ f(y) > 0 $ (solutions increase).
- For $ y > 1 $, $ 1-y < 0 $ so $ f(y) < 0 $ (solutions decrease).


<img
  src="{{ '/courses/math-methods/images/lec13/SP_Example_01_2.png' | relative_url }}"
  alt="Horizontal lines at y = 0 and y = 1. Small arrows drawn over the entire grid point towards the y = 1 line and away from the y = 0 line."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

We would say this problem has **two branches**, that is two possible equilibrium for a single value of the working variable, in this cased $x$. The bottom branch located at $y=0$ appears to be unstable as solutions on either side of this branch move away from it rather than towards it. The top branch located at $y = 1$ appear to be stable since solutions approach it from both sides
{% endcapture %}
{% include example.html content=ex %}


### Physical Significance:
In physics and engineering, qualitative analysis and slope field diagrams provide a fast, intuitive understanding of a system's dynamics. They help answer questions such as:

- What are the possible long-term behaviors of the system?
- Which states are stable, and which are unstable?
- How does the system respond to small perturbations (i.e., small nudges away from equilibrium)?

By taking a moment to use qualitative analysis and slope field diagrams, you can predict the overall behavior of complex systems even when finding an explicit solution is difficult or impossible. This approach not only deepens your conceptual understanding of differential equations but also prepares you for more advanced topics where qualitative behavior is as important as quantitative solutions.













## Bernoulli's Equation

Bernoulli's Equation represents a special class of nonlinear first-order differential equations that can be transformed into a linear ODE by an appropriate substitution. The general form of a Bernoulli equation is:

$$
\frac{dy}{dx} + p(x)y = q(x)y^n
$$

Notice, if $n=0$ or $1$, then this would be a linear ODE. Since is approach is for nonlinear ODEs we will restrict $ n \neq 0,1 $.  This restriction in conjunction with the $ y^n $ term, the equation is nonlinear. However, by applying the substitution

$$
v = y^{\ 1-n}
$$

we can reduce the equation to a linear form.






### General Outline of the Approach

**Step 1: Write the Equation in Standard Form**

We start with:

$$
\frac{dy}{dx} + p(x)y = q(x)y^n
$$

**Step 2: Perform the Substitution**

Let:

$$
v = y^{\ 1-n}
$$

Differentiate both sides with respect to $ x $ using the chain rule:

$$
\frac{dv}{dx} = (1-n) y^{-n} \frac{dy}{dx}
$$

where the $(1-n) y^{-n} $ comes from using the power rule and $\frac{dy}{dx}$ comes from the chain rule. Solve for $\frac{dy}{dx}$:

$$
\frac{dy}{dx} = \frac{1}{1-n} y^n \frac{dv}{dx}
$$

which gives us a way to rewrite the ODE in terms of $v(x)$. 

**Step 3: Substitute into the Original Equation**

Substitute $\frac{dy}{dx}$ into the Bernoulli equation:

$$
\frac{1}{1-n} y^n \frac{dv}{dx} + p(x)y = q(x)y^n
$$

Divide through by $ y^n $ (assuming $ y \neq 0 $):

$$
\frac{1}{1-n} \frac{dv}{dx} + p(x)y^{\ 1-n} = q(x)
$$

Since $ y^{\ 1-n} = v $, this becomes:

$$
\frac{1}{1-n} \frac{dv}{dx} + p(x)v = q(x).
$$

Multiply both sides by $ 1-n $ to obtain the linear ODE in $ v $:

$$
\frac{dv}{dx} + (1-n)p(x)v = (1-n)q(x)
$$

**Step 4: Solve the Linear ODE for $ v(x) $**

Now that the equation is linear in $ v $, you can solve it using the integrating factor method where $P(x) =(1-n)p(x)$ and $Q(x) = (1-n)q(x) $. 

Once you obtain $ v(x) $, you an use that to get $ y(x) $:

$$
y(x) = v(x)^{\frac{1}{1-n}}
$$


{% capture ex %}
Consider the Bernoulli equation:

$$
\frac{dy}{dx} - \frac{2}{x}y = -x^3 y^3
$$

By inspection we can see that $ p(x) = -\frac{2}{x} $, $ q(x) = -x^3 $, and $ n = 3 $. Since $ n = 3 $, the equation is nonlinear, and can use the substitution $ v = y^{1-3} = y^{-2} $ to convert this into linear form.

First, we can differentiate $ v = y^{-2} $ with respect to $ x $ to find the substitution for $\frac{dy}{dx}$:

$$
\frac{dv}{dx} = \frac{d}{dx}\left(y^{-2}\right) =  -2 y^{-3}  \frac{dy}{dx} \quad \Longrightarrow \quad \frac{dy}{dx} = -\frac{1}{2} y^3 \frac{dv}{dx}
$$

Subbing this into the original equation:

$$
\left(-\frac{1}{2} y^3 \frac{dv}{dx}\right) - \frac{2}{x}y = -x^3 y^3
$$

and simplifying by dividing the entire equation by $ y^3 $ and multiplying by $-2$:

$$
\frac{dv}{dx} + \frac{4}{x} y^{-2} = 2x^3
$$

Since $ y^{-2} = v $, this simplifies to:

$$
\frac{dv}{dx} + \frac{4}{x} \  v = 2x^3
$$

We can solve this ODE using an integrating factor with $P(x) = \frac{4}{x} $ and $Q(x) = 2 x^3 $. The integrating factor will be:

$$ \mu(x) = e^{\int P(x) \  dx } = e^{\int \frac{4}{x} \  dx } =  e^{4 \ln \vert x \vert } =  e^{\ln{x^4}}  =   x^4 \quad \Longrightarrow \quad \mu(x) = x^4    $$

Multiplying the ODE by this integrating factor and expressing the left-hand side as a product rule gives:

$$ \frac{d}{dx} \left( x^4 \  v(x) \right) = 2 x^7 $$

Integrating both sides with respect to $x$ gives:

$$  x^4 \  v(x) = \frac{1}{4} x^8 + C  $$

which we can solve for $v(x)$ to get:

$$ v(x) = \frac{1}{4} x^4 + C x^{-4}  $$

Now that we have $v(x)$, we can get $y(x)$ by using the original substitution we made: 

$$ v(x) = y(x)^{-2}  \quad \Longrightarrow \quad  y(x) = \left( v(x) \right)^{-1/2} \quad \Longrightarrow \quad  y(x) = \left( \frac{1}{4} x^4 + C x^{-4}  \right)^{-1/2}  $$

So, our general solution to the original nonlinear ODE is:

$$ y(x) = \left( \frac{1}{4} x^4 + C x^{-4}  \right)^{-1/2} $$
{% endcapture %}
{% include example.html content=ex %}




















## Variation of Parameters for First-Order ODEs

The last method we will learn for solving first-order ODEs is called **variation of parameters**. This method offers an alternative approach to finding a particular solution for nonhomogeneous, linear, first-order differential equations. Although the integrating factor method is often more straightforward, variation of parameters can provide additional insight into the structure of the solution. Also, the integrating factor method is not readily useable for higher order ODEs, whereas variation of parameters is, which is why we are introducing it here at the end of our discussion of first-order ODEs.

Here is a general overview of how this method works. Consider a first-order ODE in standard form:

$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

We begin by solving the associated homogeneous equation:

$$
\frac{dy}{dx} + P(x)y = 0
$$

The solution to the homogeneous equation can be found using separation of variables:

$$
y_h(x) = e^{-\int P(x)\ dx}
$$




### Note about Homogeneous and Particular Solutions

When solving ODEs, especially higher order ones, we will often find the homogeneous and particular solutions separately. In this case, the particular solution is **not** the one that satisfies some set of initial conditions, but it the solution to the ODE that satisfies the inhomogeneous term. This is a divide and conquer concept where we will use the inhomogeneous solution to satisfy the inhomogeneous term (i.e., the driving term) in the ODE, and we will use the homogeneous solution to take care of the initial conditions. 

For each solution we have the following:

$$
\begin{aligned}
	\text{Homogeneous: } & \frac{dy_h}{dx} + P(x)y_h = 0 \\
	\text{Particular: } & \frac{dy_p}{dx} + P(x)y_p = Q(x) 
\end{aligned}
$$

Notice if we add these two equations together we get:

$$
\begin{aligned}
	\frac{dy_h}{dx} + \frac{dy_p}{dx} + P(x)y_h + P(x)y_p  &= 0 + Q(x) \\
	\frac{d}{dx} \left(y_h + y_p\right) + P(x)\left(y_h + y_p\right)  &= Q(x) 
\end{aligned}
$$

If we make the following definition:

$$ y(x) = y_h(x) + y_p(x) $$

then we will have the general solution to the original ODE:

$$\frac{dy}{dx} + P(x) y = Q(x) $$

Notice **this trick only works for linear ODEs**! If the ODE is nonlinear, then this trick is not valid. This is part of the reason nonlinear ODEs are so hard to solve without resorting to numerical solutions. 



The idea behind variation of parameters is to look for a particular solution $ y_p(x) $ in the form:

$$
y_p(x) = u(x) \  y_h(x)
$$

where $ u(x) $ is an unknown function the we will need to find. But how do we do that? 

First we need an equation that $u(x)$ must satisfy, then we can solve that to get $u(x)$ and subsequently $y_p(x)$. 

**Step 1: Differentiate the Particular Solution** 

Differentiate $ y_p(x) = u(x)y_h(x) $ with respect to $ x $:

$$
\frac{dy_p(x)}{dx} = \frac{d}{dx}\left( u(x) \  y_h(x) \right)  =  \frac{du}{dx}y_h + u \frac{dy_h}{dx}
$$

**Step 2: Substitute into the Original ODE**
Substitute $ y_p(x) $ and $ \frac{dy_p}{dx} $ into the original ODE:

$$
\frac{dy_p}{dx} + P(x)y_p = Q(x)  \quad \implies \quad  \frac{du}{dx}y_h + u \frac{dy_h}{dx} + P(x)u y_h = Q(x)
$$

Now, notice we can group things in the following manner: 

$$
\frac{du}{dx}y_h + u \left( \frac{dy_h}{dx} + P(x)y_h \right)  = Q(x)
$$

Since the homogeneous solution satisfies

$$
\frac{dy_h}{dx} + P(x)y_h  = 0
$$

the terms involving $ u(x) $ goes to zero, leaving:

$$
\frac{du}{dx}y_h + u \left( \underbrace{\frac{dy_h}{dx} + P(x)y_h}_{=0} \right)  = Q(x) 
$$

$$
\frac{du}{dx}y_h = Q(x)
$$

**Step 3: Solve for $ u(x) $**

Solve for $ \frac{du}{dx} $:

$$
\frac{du}{dx} = \frac{Q(x)}{y_h(x)}
$$

and integrate both sides to find $ u(x) $:

$$
u(x) = \int \frac{Q(x)}{y_h(x)}\ dx
$$

**Step 4: Write the General Solution**

This means the particular solution is then given by:

$$
y_p(x) = y_h(x) \int \frac{Q(x)}{y_h(x)}\ dx
$$

Thus, the general solution to the original ODE is:

$$
\begin{aligned}
	y(x) &= y_h(x) + y_p(x) \\
	&= y_h(x) + y_h(x) \int \frac{Q(x)}{y_h(x)}\ dx  \\
	y(x) &=  y_h(x) \left[ C + \int \frac{Q(x)}{y_h(x)}\ dx \right]
\end{aligned}
$$

where $ C $ is the constant of integration.

**Discussion:**

While this approach yields the same solution as the integrating factor method for a first order ODE, it offers a general approach for getting particular solution for inhomogeneous ODEs by explicitly modifying the homogeneous solution with the function $ u(x) $. This method is particularly useful when trying to solve second-order, inhomogeneous ODEs.

{% capture ex %}
Consider the differential equation

$$
\frac{dy}{dx} + 2y = e^{-x}
$$

with an initial condition $ y(0) = y_0 $. 

The homogeneous differential equation to this problem is given as :

$$
\frac{dy_h}{dx} + 2y_h = 0.
$$

Its general solution is found by separation of variables or by inspection:

$$
y_h(x) = Ce^{-2x}
$$

where $ C $ is an arbitrary constant.

We now look for a particular solution $ y_p(x) $ in the form:

$$
y_p(x) = u(x)y_h(x) = u(x)e^{-2x}
$$

where $ u(x) $ is an unknown function to be determined. This particular solution must solve the following inhomogeneous ODE:

$$
\frac{dy_p}{dx} + 2y_p = e^{-x}
$$

Let's first find the first derivative of the particular solution we have assumed:

$$
\frac{dy_p}{dx} =  \frac{du}{dx} \   e^{-2x} - 2u(x)e^{-2x}
$$

and then substitute the particular solution into the inhomogeneous ODE:

$$
\frac{dy_p}{dx} + 2y_p = e^{-x} \qquad \implies \qquad 	\frac{du}{dx} \   e^{-2x} - 2u(x)e^{-2x}  + 2u(x)e^{-2x}  = e^{-x} 
$$

Notice the $u(x)$ terms cancel out, leaving us with:

$$ \frac{du}{dx} \   e^{-2x} = e^{-x}  $$ 

which can be rearranged to get:

$$ \frac{du}{dx} = e^{x}  $$ 

Now we can integrate with respect to $ x $ to find an equation for $u(x)$:

$$
u(x) = \int e^{x}\ dx = e^{x} + K
$$

where $ K $ is a constant of integration. For a particular solution, we can set $ K = 0 $ (since any constant here would be absorbed into the homogeneous solution--check this is your are curious). 

In the end, we have:

$$
u(x) = e^{x}
$$

and substituting this into $ y_p(x) $:

$$
y_p(x) = u(x)e^{-2x} = e^{x}e^{-2x} = e^{-x}
$$

Therefore, the general solution to the ODE is the sum of the homogeneous and particular solutions:

$$
\begin{aligned}
    y(x) &= y_h(x) + y_p(x)\\
    y(x) &= Ce^{-2x} + e^{-x}
\end{aligned}
$$

We can apply the initial condition to get:

$$ y(0) = y_0 =  C + 1 \quad \implies \quad C = (y_0 - 1)$$

Putting this back into the solution gives:

$$ y(x) = (y_0 - 1)e^{-2x} + e^{-x}$$ 

which is our solution to the initial value problem.
{% endcapture %}
{% include example.html content=ex %}










## Additional Applications


### Example 1: Slope Field Diagram

Consider the autonomous ODE:

$$
\frac{dy}{dx} = y^2 - 4x + 3
$$

This is a mildly complicated first-order, nonlinear, inhomogeneous, ODE that would be a pain to solve by hand. Instead, let's get some understanding of the ODE by creating and dissecting its slope field diagram. First, notice from the way this is written we can identify $f(x,y)$ as:

$$ f(x,y) = y^2 - 4x + 3 $$

Setting this function to zero will give us our equilibrium lines (aka nullclines):

$$
y^2 - 4x + 3 =0 \quad \implies \quad y=\pm \sqrt{4 x - 3} 
$$

Plotting this

<img
  src="{{ '/courses/math-methods/images/lec13/SP_Example_02_1.png' | relative_url }}"
  alt="A parabola opening to the right."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

we see that it has two branches. This nullcline breaks the plot into two regions, to the left of the nullcline and to the right of it. If we examine the sign of $f(x,y)$ in each of these regions we get:

- To the left of the nullcline: Choose $x = 0$ and $ y = 0$; then, $0 - 4(0) + 3 = 3 > 0$, we get positive slopes.
- To the right of the nullcline: Choose $x = 2$ and $ y = 0$; then, $0 - 4(2) + 3 = -5 < 0$, we get negative slopes.

Adding in the slopes gives the following plot:

<img
  src="{{ '/courses/math-methods/images/lec13/SP_Example_02_2.png' | relative_url }}"
  alt="A parabola opening to the right. Small arrows placed all over the the grid point away from the top half of the parabola and towards the bottom half of the parabola."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

This suggests the top branch is unstable and the bottom branch is table. So, most solutions will approach $y\rightarrow \infty$ or  $y= - \sqrt{4 x - 3} $ depending on their initial conditions. 






### Example 2: Bernoulli's Equation

Consider the Bernoulli equation:

$$
\frac{dy}{dx} + \frac{2}{x} y = x^3 y^3
$$

Here, $p(x)=\frac{2}{x}$, $q(x)=x^3$, and $n=3$. Since $n \neq 0,1$, the equation is nonlinear, but we can reduce it to a linear ODE via a substitution. Making the following change of functions

$$
v = y^{1-n} = y^{1-3} = y^{-2}
$$

and differentiating both sides with respect to $x$:

$$
\frac{dv}{dx} = -2 y^{-3} \frac{dy}{dx} \quad \implies \quad \frac{dy}{dx} = -\frac{1}{2} y^3 \frac{dv}{dx}
$$

gives us the needed substitutions to change this from a nonlinear ODE to a linear one. 

Putting these substitutions in gives:

$$
-\frac{1}{2} y^3 \frac{dv}{dx} + \frac{2}{x} y = x^3 y^3
$$

Divide the entire equation by $ y^3 $ and multiply by $-2$:

$$
\frac{dv}{dx} - \frac{4}{x} y^{-2} = -2 x^3
$$

Since $ y^{-2} = v $, this becomes:

$$
\frac{dv}{dx} - \frac{4}{x} \  v = -2 x^3
$$

We can solve this ODE using an integrating factor. The integrating factor  for $v$ is:

$$
\mu(x) = e^{\int -\frac{4}{x}\ dx} = e^{-4\ln \vert x \vert } = x^{-4}
$$

Multiply the ODE by $ x^{-4} $:

$$
x^{-4} \frac{dv}{dx} - \frac{4}{x} x^{-4} v =  -2x^{-1}
$$

The left-hand side can be rewritten using the product rule to become:

$$
\frac{d}{dx}\Bigl(x^{-4} v\Bigr) = -2x^{-1}
$$

which we can integrate to get:

$$
x^{-4} v =  -2\ln \vert x \vert  + C
$$

where $C$ is the unknown constant of integration. 

Solving this for $v(x)$:

$$
v(x) = x^4 \left( -2\ln \vert x \vert  + C \right)
$$

and reverting this back to $y(x)$ using the fact $ v = y^{-2} $, so:

$$
y^{-2} = x^4 \left( -2\ln \vert x \vert  + C \right)
$$

and hence, the general solution for $ y(x) $ is:

$$
y(x) = \frac{1}{\sqrt{x^4 \left( -2\ln \vert x \vert  + C \right)}}
$$







### Example 3: Variation of Parameters for a First-Order ODE

Consider the nonhomogeneous ODE:

$$
\frac{dy}{dx} + 3y = 2x
$$

with the initial condition $ y(0)=y_0 $. 

Let's first get the homogeneous solution by solving the folloowing ODE:

$$
\frac{dy}{dx} + 3y = 0
$$

This can be solved using separation of variables to get:

$$
y_h(x) = Ce^{-3x}
$$

where $ C $ is an arbitrary constant.

From this homogeneous solution, we can propose out particular solutions in the following manner:

$$
y_p(x) = u(x) \  y_h(x) = u(x)e^{-3x}
$$

where $ u(x) $ is an unknown function.

We need to sub this particular solution into the inhomogeneous ODE:

$$
\frac{dy_p}{dx} + 3y_p = 2x
$$

to find an equation for $u(x)$.

First, we differentiate $ y_p(x) $:

$$
\frac{dy_p}{dx} = \frac{du}{dx} \  e^{-3x} - 3u(x)e^{-3x}
$$

and then substitute in this and $ y_p(x) $:

$$
\left[\frac{du}{dx} \  e^{-3x} - 3u(x)e^{-3x}\right] + 3\left[u(x)e^{-3x}\right] = 2x
$$

The $ -3u(x)e^{-3x} $ and $ +3u(x)e^{-3x} $ terms cancel, leaving:

$$
\frac{du}{dx} e^{-3x} = 2x
$$

Solve for $ \frac{du}{dx} $:

$$
\frac{du}{dx}  = 2x e^{3x}
$$

and integration to get $u(x)$:

$$
u(x) = \int 2x e^{3x}\ dx
$$

We can complete this integration using integration by parts. Letting:

$$
U = 2x \quad\text{and}\quad dV = e^{3x}dx \quad \implies \quad  dU = 2dx \quad\text{and}\quad V = \frac{1}{3}e^{3x}
$$

Then,

$$
\begin{aligned}
	u(x) &= \int 2x e^{3x}\ dx \\
	&= 2x\cdot\frac{1}{3}e^{3x} - \int \frac{1}{3}e^{3x}\cdot 2dx \\
	&= \frac{2x}{3}e^{3x} - \frac{2}{3}\int e^{3x}dx  \\
	&= \frac{2x}{3}e^{3x} - \frac{2}{9} e^{3x}
\end{aligned}
$$ 

Thus, we can write:

$$
u(x) = e^{3x}\left(\frac{2x}{3} - \frac{2}{9}\right)
$$

Now that we have $u(x)$ we can get the particular solution:

$$
y_p(x) = u(x)e^{-3x} = \left(e^{3x}\left(\frac{2x}{3} - \frac{2}{9}\right)\right)e^{-3x} = \frac{2x}{3} - \frac{2}{9}
$$

and the general solution as the sum of the homogeneous and particular solutions:

$$
\begin{aligned}
	y(x) &= y_h(x) + y_p(x) \\
	y(x) &= Ce^{-3x} + \frac{2x}{3} - \frac{2}{9} \\
\end{aligned}
$$



Applying the given initial condition:

$$ y(0) = y_0 = C - \frac{2}{9} \quad\implies\quad C = y_0 + \frac{2}{9} $$ 

This gives:

$$y(x) = \left(y_0 + \frac{2}{9}\right)e^{-3x} + \frac{2x}{3} - \frac{2}{9} $$

















## Problem:

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


Consider the differential equation

$$
\frac{dy}{dx} + \frac{3}{x}\ y = x^2\  y^2
$$

which might arise in a simplified chemical reaction model.

a) **Bernoulli's Equation**
	
Show that this ODE is of Bernoulli type with $ n = 2 $. Use the substitution

$$
v = y^{\ 1-n} = y^{-1}
$$

and express the ODE in terms of $ v(x) $. Solve the resulting linear ODE to find the general solution for $ y(x) $.

b) **Particular Solution** 

Given the initial condition

$$
y(1) = 1
$$

find the particular solution.
	
c) **Qualitative Analysis**

Perform a qualitative analysis by:

- Identifying the equilibrium lines and sketching them in a plot.

- Discussing the stability of the equilibrium lines you found.



**Hint:** Recall that a Bernoulli equation of the form 
$$
\frac{dy}{dx} + p(x)y = q(x)y^n
$$
can be linearized by the substitution $ v = y^{1-n} $, after which standard techniques (such as the integrating factor method) can be used to solve for $ v(x) $ and then recover $ y(x) $.