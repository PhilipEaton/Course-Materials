---
layout: default
title: Mathematical Methods - Lecture 13
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 13
---



# Lecture 13 – Advanced Topics in First-Order ODEs


## Introduction

In our previous lectures, we discussed the basic methods for solving first-order ordinary differential equations; specifically, separation of variables and the integrating factor method. These techniques served us well in tackling a variety of linear ODEs in contexts ranging from RC circuits to disease modeling. However, as we advance our study of differential equations, we will encounter cases and models that require more sophisticated approaches. 

Let's expand our analytical toolkit by exploring a few more advanced methods useful for solving first-order ODEs. Our goal is to handle a broader class of equations and gain a deeper conceptual understanding of their behavior in physical systems. The topics we will cover include:

- **Slope Field Diagrams:** helpful for understanding the long-term behavior of ODEs without needing to find an explicit solution.
- **Bernoulli's Equation:** A special type of nonlinear ODE that, through a clever substitution, can be reduced to a solvable linear ODE.
- **Variation of Parameters for First-Order ODEs:** An alternative method for solving inhomogeneous equations, complementing the integrating factor approach. We will see this method expanded upon when we go to solve second order ODEs.

We will practice these methods by solving ODEs from population dynamics, chemical kinetics, and economics that illustrate the diverse applicability of first-order ODEs.








## Qualitative Analysis and Slope Field Diagrams

Not all differential equations yield neat, closed-form solutions. In many cases, however, we do not need an explicit solution to understand the behavior of the system being modeled. Instead, qualitative analysis can provide valuable insights into the dynamics of a system. One particularly useful tool in qualitative analysis is the slope field diagram.

Qualitative analysis involves using *logical reasoning* and an *understanding of the physical context* to predict the general behavior of solutions without having to solve the ODE. In other words, we use intuition and deductions about the system to anticipate features such as equilibrium points, stability, and the overall direction of change.

For example, suppose you are studying an ODE that models the motion of a mass attached to a spring. From experience, you know that when the mass is set into motion it will oscillate. This means you can expect the solution to involve oscillatory behavior, often represented by functions like sine and cosine. we do not need to solve the ODE to come to this conclusion, we know it from our physical intuition. When we actually go to solve the problem, if we do not get oscillatory functions, then we should probably double check our work. It doesn't mean our answer is right or wrong, it give us the ability to "sanity check" our answers. 

In addition to this kind of physical reasoning, we can examine mathematical features of the equation itself. In particular, equilibrium points (or nullclines), their stability, and the direction of change of the solution can tell us a great deal about how the system evolves over time. These features can be visualized using a slope field diagram.

As we have been emphasizing, slope field diagrams are useful tools for examining first-order ODEs. However, we have not yet formally defined what they are. Let’s correct that now.

Consider a general first-order ODE:

$$
\frac{dy}{dt} = f(y,t)
$$

Equilibrium (or steady-state) solutions occur when the solution no longer changes, that is when:

$$
\frac{dy}{dt} = f(y,t) = 0
$$

Solving this equation gives the values for $y$ (possibly as functions of $t$, the working variable) where the solution does not change. We have been caling these solutions the **steady-state solutions**. In mathematics, the curves defined by this condition are called **nullclines**. Nullclines represent equilibrium points or equilibrium curves of the ODE.

To understand the stability of these equilibria, we examine the behavior of the solution near the nullcline. In the neighborhood of a nullcline:

- If $f(y,t) > 0$, then $\frac{dy}{dt} > 0$, meaning the solution increases with time.
- If $f(y,t) < 0$, then $\frac{dy}{dt} < 0$, meaning the solution decreases with time.

To see how useful this is, consider, for example, if the slope $\frac{dy}{dt}$ is positive below the nullcline. This means solutions in that region would move upward toward the nullcline. If the slope were negative instead, solutions below the nullcline would move downward, away from it. The same reasoning can be applied to the region above the nullcline.

If solutions on both sides of the nullcline move toward it, the equilibrium is stable. If solutions move away from the nullcline on either side, the equilibrium is unstable.

These behaviors become much easier to interpret when we visualize them using a slope field. A slope field is a two-dimensional diagram where short line segments represent the slope $\frac{dy}{dt}$ at various points in the $(t,y)$ plane. The nullclines divide the diagram into regions, and the slopes in each region indicate whether $y$ increases or decreases as time progresses.

### Constructing a Slope Field Diagram

To construct a slope field diagram, follow these steps:

- Identify the nullclines by solving $f(y,t) = 0$.
- Plot the nullclines on a $y$ versus $t$ diagram.
- Determine the sign of $f(y,t)$ in the regions between the nullclines.
- Draw small line segments indicating the slope $\frac{dy}{dt}$ at representative points in each region.


{% capture ex %}
Consider the ODE:

$$
\frac{dy}{dt} = y(1-y)
$$

For this ODE we have $f(y,t) = y(1-y)$. The equilibrium points/lines can be found by setting the derivative equal to zero and solving for $y$. This is equivalent to finding the steady-state solutions:

$$
y(1-y) = 0 \qquad \implies \qquad y = 0 \quad \text{and} \quad y = 1
$$

Drawing these nullclines in a $y$-versus-$t$ coordinate system gives:

<img
src="{{ '/courses/math-methods/images/lec13/SP_Example_01_1.png' | relative_url }}"
alt="Horizontal lines at y = 0 and y = 1."
style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

Next we test the sign of the slope, $f(y) = y(1-y)$, above and below each of the nullclines. This can be done by inspection, or by selecting representative values in each region and checking the sign of the result.

- For $y < 0$:

  Let’s choose $y = -1$. Then

  $$
  f(-1) = (-1)\big(1-(-1)\big) = (-1)(2) = -2
  $$

  So $f(y) < 0$ in this region.

- For $0 < y < 1$:

  We can use $y = 0.5$. This gives

  $$
  f(0.5) = (0.5)\big(1-0.5\big) = (0.5)(0.5) = 0.25
  $$

  So $f(y) > 0$ in this region.

- For $y > 1$:

  Let’s choose $y = 2$. Then

  $$
  f(2) = (2)\big(1-2\big) = (2)(-1) = -2
  $$

  So $f(y) < 0$ in this region.

Plotting the slopes for each region gives the following slope field:

<img
  src="{{ '/courses/math-methods/images/lec13/SP_Example_01_2.png' | relative_url }}"
  alt="Horizontal lines at y = 0 and y = 1. Small arrows drawn over the grid point toward the y = 1 line and away from the y = 0 line."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

We say this problem has **two branches**, that is two possible equilibria for a single value of the working variable $t$. The bottom branch located at $y=0$ appears to be unstable as solutions on either side of this branch move away from it rather than towards it. The top branch located at $y = 1$ appears to be stable since solutions approach it from both sides.

Where our solution ultimately ends up depends on the initial condition.  If the initial condition satisfies $y > 0$, the solution will move toward the equilibrium at $y = 1$ and eventually converge to that value. However, if the initial condition lies in the region $y < 0$, the solution moves downward and continues decreasing without bound, approaching $-\infty$ as time progresses.

Thus, the long-term behavior of the system described by this ODE depends on the initial condition. For initial values above zero, the solution approaches the stable equilibrium at $y = 1$, these are stable. For initial values below zero, the solution diverges away from the system and decreases without bound, these are unstable.
{% endcapture %}
{% include example.html content=ex %}



### Physical Significance

In physics and engineering, qualitative analysis and slope field diagrams provide a fast and intuitive way to understand a system's dynamics. They help answer questions such as:

- What are the possible long-term, steady-state behaviors of the system?
- Which states are stable and which are unstable?
- How does the system respond to small perturbations (that is, small nudges away from equilibrium)?

As we just saw in the previous example, we were able to determine how solutions behave over time without ever solving the differential equation explicitly. By examining the nullclines and the sign of the slope in each region, we could predict which equilibria the system moves toward and which it moves away from.

Taking a moment to perform this kind of qualitative analysis often provides a clear picture of the system’s overall behavior, even when finding an analytic solution (a closed-form function) is difficult or impossible. This approach not only deepens our conceptual understanding of differential equations, but it also prepares us for more advanced topics where qualitative behavior is just as important as exact solutions.







## Bernoulli's Equation

While qualitative analysis helps us understand the general behavior of solutions, there are still many nonlinear differential equations that we would like to solve explicitly. One important example is Bernoulli's equation.

Bernoulli's equation represents a special class of nonlinear first-order differential equations that can be transformed into a linear ODE through an appropriate substitution. The general form of a Bernoulli equation is:

$$
\frac{dy}{dx} + p(x)y = q(x)y^n
$$

Notice if $n = 0$ or $n = 1$, the equation becomes linear. 

Bernoulli's method is designed for nonlinear equations, so we will restrict our attention to the case where $n \neq 0, 1$. With this restriction, the presence of the $y^n$ term makes the equation nonlinear.

It turns out this ODE can be transformed into a linear ODE, as if by magic, by making the following substitution:

$$
v = y^{\,1-n}
$$

After applying this substitution, the original differential equation can be rewritten as a linear ODE in terms of the new function $v$ that depending on the working variable $x$.




### General Outline of the Approach

**Step 1: Write the Equation in Standard Form**

We begin by rewriting the differential equation in the form

$$
\frac{dy}{dx} + p(x)y = q(x)y^n
$$


**Step 2: Perform the Substitution**

If the equation can be written in this form, then it is a Bernoulli equation. Once identified, it can be simplified, though not directly solved, by applying the following substitution:

$$
v = y^{\ 1-n}
$$

To find how this simplifies the ODE, we first differentiate both sides with respect to $ x $:

$$
\frac{dv}{dx} = (1-n) y^{-n} \frac{dy}{dx}
$$

where the $(1-n) y^{-n} $ comes from using the power rule and $\frac{dy}{dx}$ comes from the chain rule. Solving for $\frac{dy}{dx}$:

$$
\frac{dy}{dx} = \frac{1}{1-n} y^n \frac{dv}{dx}
$$

gives us a way to rewrite the ODE in terms of $v(x)$. 

**Step 3: Substitute into the Original Equation**

Substitute $\frac{dy}{dx}$ into the original ODE:

$$
\frac{1}{1-n} y^n \frac{dv}{dx} + p(x)y = q(x)y^n
$$

and divide through by $ y^n $ (assuming $ y \neq 0 $):

$$
\frac{1}{1-n} \frac{dv}{dx} + p(x)y^{\ 1-n} = q(x)
$$

Since $ y^{\ 1-n} = v $, this becomes:

$$
\frac{1}{1-n} \frac{dv}{dx} + p(x)v = q(x)
$$

Finally, we can multiply both sides by $ 1-n $ to obtain a linear ODE for $ v $ in standard form:

$$
\frac{dv}{dx} + (1-n)p(x)\ v = (1-n)q(x)
$$

**Step 4: Solve the Linear ODE for $ v(x) $**

Now that the equation is linear in $ v $, we can solve it using the integrating factor method where $P(x) =(1-n)p(x)$ and $Q(x) = (1-n)q(x) $. 

Once we obtain $ v(x) $, it can use that to find $ y(x) $:

$$
y(x) = v(x)^{\frac{1}{1-n}}
$$

Let's put this procedure into practice by looking at an example.

{% capture ex %}
Consider the following differential equation:

$$
\frac{dy}{dx} - \frac{2}{x}y = -x^3 y^3
$$

By inspection we can see that this is in the general form for a Bernoulli Equation. Here we have $ p(x) = -\frac{2}{x} $, $ q(x) = -x^3 $, and $ n = 3 $. Since $ n = 3 $, the equation is nonlinear, and can use the substitution $ v = y^{1-3} = y^{-2} $ to convert this into linear form.

First, we can differentiate $ v = y^{-2} $ with respect to $ x $ to find the substitution for $\frac{dy}{dx}$:

$$
\frac{dv}{dx} = \frac{d}{dx}\left(y^{-2}\right) =  -2 y^{-3}  \frac{dy}{dx} \quad \Longrightarrow \quad \frac{dy}{dx} = -\frac{1}{2} y^3 \frac{dv}{dx}
$$

Subbing this into the original equation:

$$
\left(-\frac{1}{2} y^3 \frac{dv}{dx}\right) - \frac{2}{x}y = -x^3 y^3
$$

and, following the process, we divide the entire equation by $ y^3 $ and multiply by $-2$:

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

$$ \frac{d}{dx} \left( x^4 \  v \right) = 2 x^7 $$

Integrating both sides with respect to $x$ gives:

$$  x^4 \  v = \frac{1}{4} x^8 + C  $$

which we can solve for $v(x)$ to get:

$$ v(x) = \frac{1}{4} x^4 + C x^{-4}  $$

Armed with $v(x)$, we can find $y(x)$ by using the original substitution we made: 

$$ v(x) = y(x)^{-2}  \quad \Longrightarrow \quad  y(x) = \left( v(x) \right)^{-1/2} \quad \Longrightarrow \quad  y(x) = \left( \frac{1}{4} x^4 + C x^{-4}  \right)^{-1/2}  $$

Our general solution to the original nonlinear ODE is:

$$ y(x) = \left( \frac{1}{4} x^4 + C x^{-4}  \right)^{-1/2} $$
{% endcapture %}
{% include example.html content=ex %}












## Variation of Parameters for First-Order ODEs

The last method we will learn for solving first-order ODEs is called variation of parameters. This method provides an alternative way to find a particular solution to a nonhomogeneous, linear, first-order differential equation.

Although the integrating factor method is often more straightforward, variation of parameters can offer additional insight into the structure of the solution. More importantly, while the integrating factor method does not generalize easily to higher-order ODEs, variation of parameters does. Because of this, it is a method worth understanding early.

Consider a first-order ODE written in standard form:

$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

We begin by solving the associated homogeneous equation:

$$
\frac{dy}{dx} + P(x)y = 0
$$

This equation can be solved using separation of variables, giving the homogeneous solution

$$
y_h(x) = e^{-\int P(x)\,dx}
$$


### Note about Homogeneous and Particular Solutions

When solving ODEs, especially higher-order ones, it is common to find the homogeneous and particular solutions separately. In this context, the particular solution is not the solution that satisfies a specific set of initial conditions. Instead, it is the solution that accounts for the inhomogeneous (or forcing) term in the differential equation.

You can think of this as a divide-and-conquer strategy. The particular solution handles the inhomogeneous or driving term in the ODE, while the homogeneous solution is used to incorporate the initial conditions.

Unfortunately, the word “particular” is used in two slightly different ways when discussing differential equations. In one sense, it refers to the final solution obtained after applying initial conditions, making the solution particular to the specific physical situation. In another sense, it refers to the solution associated with the inhomogeneous term of the differential equation. Context usually makes it clear which meaning is intended.

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

The central idea behind variation of parameters is to look for a particular solution $y_p(x)$ that has a form similar to the homogeneous solution, but with a coefficient that is allowed to vary with $x$. Specifically, we assume:

$$
y_p(x) = u(x) \ y_h(x)
$$

where $y_h(x)$ is the solution to the homogeneous equation and $u(x)$ is an unknown function that we will determine.

Why does this idea make sense and why does this work?

Recall that the homogeneous solution $y_h(x)$ already captures the natural behavior of the system when no external forcing is present. In the homogeneous equation, this solution can be multiplied by any constant and still remain a valid solution. That is, if $y_h(x)$ is a solution, then $C\,y_h(x)$ is also a solution for any constant $C$.

The key insight of variation of parameters is to relax this idea slightly. Instead of multiplying the homogeneous solution by a constant, we allow the coefficient to vary with $x$, replacing $C$ with $u(x)$.

By allowing this coefficient to change with $x$, the solution gains enough flexibility to account for the inhomogeneous term in the differential equation. Intuitively, you can think of the function $u(x)$ as adjusting the amplitude of the homogeneous solution at each value of $x$ so that the resulting function satisfies the full inhomogeneous equation.

In this sense, we are still using the natural structure of the homogeneous solution, but we allow its scaling to vary continuously so that it can correctly respond to the forcing term in the equation. The goal of the method is therefore to determine the function $u(x)$ that makes $y_p(x) = u(x) \ y_h(x)$ satisfy the original differential equation.

We need to find the equation $u(x)$ must satisfy so can solve that to get $u(x)$ and subsequently $y_p(x)$. 

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

We can group terms based on how they depend on $u$ in the following manner: 

$$
\frac{du}{dx}y_h + u \left( \frac{dy_h}{dx} + P(x)y_h \right)  = Q(x)
$$

However, the homogeneous solution satisfies

$$
\frac{dy_h}{dx} + P(x)y_h  = 0
$$

which means the the term involving $ u(x) $ goes to zero. This leaves us with:

$$
\frac{du}{dx}y_h = Q(x)
$$

**Step 3: Solve for $ u(x) $**

This is a separable ODE! We can solve for $ \frac{du}{dx} $:

$$
\frac{du}{dx} = \frac{Q(x)}{y_h(x)}
$$

and integrate both sides with respect to $dx$ to find $ u(x) $:

$$
u(x) = \int \frac{Q(x)}{y_h(x)}\ dx
$$

We will ignore the constant of integration here since we are looking for a solution to specifically satisfy the inhomogeneous term of the original ODE and nothing more. Any unknown constants we need to satisfy initial conditions will come from the homogeneous solution; divide and conquer.

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

where the constant of integration $C$ was implicitly contained within the homogeneous solution.

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
\frac{dy_h}{dx} + 2y_h = 0
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

Notice we did not include the constant $C$ here. This is, again, because the particular solution is only here to satisfy the inhomogeneous term of the original ODE and nothing more.

We can substitute this particular solution into the inhomogeneous ODE to get:

$$
\frac{dy_p}{dx} + 2y_p = e^{-x} \qquad \implies \qquad 	\frac{du}{dx} \   e^{-2x} - 2u(x)e^{-2x}  + 2u(x)e^{-2x}  = e^{-x} 
$$

Notice the $u(x)$ terms cancel out, leaving us with:

$$ \frac{du}{dx} \   e^{-2x} = e^{-x}  $$ 

which can be rearranged to get:

$$ \frac{du}{dx} = e^{x}  $$ 

Now we can integrate with respect to $ x $ to find an equation for $u(x)$:

$$
u(x) = \int e^{x}\ dx = e^{x}
$$

Substituting this into $ y_p(x) $:

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

Consider the following ODE:

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

we see the solution has two branches (two options for a single x value, the top and bottom branches). This nullcline breaks the plot into two regions, to the left of the nullcline and to the right of it. If we examine the sign of $f(x,y)$ in each of these regions we get:

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

Here, $p(x)=\frac{2}{x}$, $q(x)=x^3$, and $n=3$. Since $n \neq 0$ or $1$ the equation is nonlinear, but we can transform it to a linear ODE via the Bernoulli substitution. Making the following change of functions:

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