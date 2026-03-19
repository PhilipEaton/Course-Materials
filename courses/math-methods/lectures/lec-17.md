---
layout: default
title: Mathematical Methods - Lecture 17
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 17
---

# Lecture 17 – Laplace Transformations I



## Introduction: The What and Why of Laplace Transforms

Up until now, we’ve tackled differential equations using different direct solution techniques such as the methods of separation of variables, integrating factors, characteristics (guess an exponential), undetermined coefficients, and variation of parameters, to name a few. However, sometimes solving an ODE directly isn’t the easiest approach. 

**Suppose we could transform the differential equation into an algebraic equation**—one we can solve just like a regular algebra problem—instead of wrestling with all the derivatives and multiple solution steps. But what kind of transformation would we do? Turns out there are a multitude of options, but the most popular by far are the Laplace and Fourier Transformations.

**Laplace and Fourier Transformations** convert differential equations into algebraic problems, making solving a differential equation significantly easier. Here we will discuss the Laplace Transformation, and the Fourier Transformation will be handled in a later chapter when we have more familiarity with functional orthogonality and other related topics.

### The What: Laplace Transformation

The **Laplace transform** takes a function $ f(t) $ whose working variable is $t$ (depending on what $t$ represents this could be time, space, etc.), and transforms it into a function $ F(s) $, a function of $s$, the Laplace domain (which could be frequency, wave-number, etc.\ depending on what $t$ represents):

$$
\mathcal{L}\{f(t)\}(s) = F(s) = \int\limits_0^\infty e^{-st} f(t) \, dt
$$

Notice the working variable $t$ is integrated over a set of definite limits. This means there will be no $t$'s in the result of this integration. This is why we say this transformations shifts us from the $t$ domain (or representation) to the $s$ domain (or representation). 


In general $ s $ can be a complex variable, but we will take it to be strictly real for this discussion. Under this assumption, something we will need to pay attention to is ensuring this integration converges for whichever function $f(t)$ is being transformed. This can typically be done  by restricting the domain of $s$ (i.e. the range of values $s$ can take) such that the integration in the Laplace Transformation will converge. Let's consider an example of how the domain of $s$ can be restricted to ensure convergence:


{% capture ex %}
Let's  calculate the Laplace transformation of a simple exponential function: $f(t) = e^{b t}$. This, by definition, can be found via the following integration:

$$ \mathcal{L}\{f(t)\}(s) = F(s) = \int\limits_{0}^{\infty} e^{-s t} e^{b t} \, dt = \int\limits_{0}^{\infty} e^{-(s-b) t} \, dt $$

Notice, for this integration to converge (i.e. to not blow up to infinity) we need the exponential to decay to zero. This requires the condition:

$$-(s - b) <0 \quad\implies\quad s - b > 0 \quad\implies\quad  s > b$$ 

This restriction on the allowed values of $s$ means the only range of values $s$ can take in the resulting function $F(s)$ are ones where $s$ is larger than $b$, whatever $b$ may be. But why is this?

We can readily understand why $s=b$ does **not** work (you cannot divide by zero), but why can't we use values less than $b$? This has to do with the definition of the transformation itself. Since the integration in the transformation is defined from $0$ to $\infty$ that cuts out the values of $s$ less than $b$. If integration in the transformation were instead defined to go from $-\infty$ to $0$, then we would find that $s$ would need to be less than $b$ and $s$ greater than $b$ would not be allowed due to the definition of the transformation. An added complication to this is that the limits of the Laplace Transformation can change depending on what you are doing, so it pays to be careful.

Completing this integration gives:

$$
\begin{aligned}
F(s)&= \mathcal{L}\{f(t)\}(s) \\[1.15ex]
	&= \int\limits_{0}^{\infty} e^{-(s-b) t} \, dt \\[1.15ex]
	&= \frac{-1}{s -b} \left.\left( e^{-(s-b)t}\right)\right|_{0}^{\infty} \, dt \\[1.15ex]
	&= \frac{-1}{s -b} \left( 0 - e^{-(s-b)0}\right) \, dt \\[1.15ex]
	&= \frac{-1}{s -b} \left( - 1\right) \\[1.15ex]
F(s)&= \frac{1}{s -b}
\end{aligned}
$$


So, the Laplace Transformation of an exponential $e^{bt}$ if given as this fraction, where $s > b$. This means, for example, if $b=5$ then:

$$ f(t) = e^{5t} \qquad \longrightarrow \qquad  F(s) = \frac{1}{s -5}$$

where $s$ can only take on values larger than $5$. 
{% endcapture %}
{% include example.html content=ex %}


Moral of the story from this example: **Be careful about the allowed domain of the resulting transformation and pay attention to how the transformation is defined.**



### Table of Common Laplace Transforms

To streamline calculations, we often use a reference table of common transforms. Here are a few that will frequently appear:

| Given $f(t)$ | Transforms to $F(s) = \mathcal{L}\{f(t)\}$ |
|---|---|
| $\frac{d}{dt} f(t)$ | $sF(s) - f(0)$ |
| $\frac{d^2}{dt^2} f(t)$ | $s^2 F(s) - s f(0) - f'(0)$ |
|  |  |
| $1$ | $\frac{1}{s}$ \quad ($s > 0$) |
| $t^n$ | $\frac{n!}{s^{n+1}}$ \quad ($s > 0$) |
| $e^{bt}$ | $\frac{1}{s - b}$ \quad ($s > b$) |
| $t^n e^{bt}$ | $\frac{n!}{(s - b)^{n+1}}$ \quad ($s > b$) |
| $\frac{1}{(s - b)^n}$ | $\frac{t^{n-1} e^{bt}}{(n-1)!}$ \quad ($n \geq 1$) |
|  |  |
| $\cos(bt)$ | $\frac{s}{s^2 + b^2}$ \quad ($s > 0$) |
| $\sin(bt)$ | $\frac{b}{s^2 + b^2}$ \quad ($s > 0$) |
| $\sinh(bt)$ | $\frac{b}{s^2 - b^2}$ \quad ($s > |b|$) |
| $\cosh(bt)$ | $\frac{s}{s^2 - b^2}$ \quad ($s > |b|$) |
| $e^{-at}\sin(bt)$ | $\frac{b}{(s + a)^2 + b^2}$ \quad ($s > -a$) |
| $e^{-at}\cos(bt)$ | $\frac{s + a}{(s + a)^2 + b^2}$ \quad ($s > -a$) |
| $\frac{\sin(bt)}{t}$ | $\tan^{-1}(b/s)$ \quad ($s > 0$) |
|  |  |
| $\frac{1}{\sqrt{t}}$ | $\sqrt{\frac{\pi}{s}}$ \quad ($s > 0$) |
| $\frac{t^m}{\Gamma(m+1)}$ | $\frac{1}{s^{m+1}}$ \quad ($s > 0$, $m > -1$) |
| $\frac{1 - e^{-at}}{t}$ | $\ln\left(\frac{s+a}{s}\right)$ \quad ($s > -a$) |
| $\frac{e^{-at}}{t}$ | $-\text{Ei}(-as)$ |
|  |  |
| $U(t-c)$ (unit step at $t=c$) | $\frac{e^{-cs}}{s}$ |
| $U(t-c)f(t-c)$ (shifted function) | $e^{-cs}F(s)$ |
| $t\,U(t-c)$ (ramp function) | $\frac{e^{-cs}}{s^2}$ |
| $\delta(t-c)$ (Dirac delta) | $e^{-cs}$ |







### The Why: Changing ODEs to Algebra Problems

We made a bold claim in the introduction to this Lecture--the Laplace Transformation can change solving an ODE to solving an algebra problem. Let's see how this can be done.

A special property of the Laplace transformation is how it interacts with derivatives. This is difficult to put into words, to let's just jump into calculating the Laplace transformation of the first derivative of a function:

$$ \mathcal{L}\left\{\frac{d f}{dt}\right\}(s) = \int\limits_{0}^{\infty} e^{-s t} \frac{d f}{dt} \, dt $$

We can strip that derivative off of the function $f(t)$ using integration by parts. Recall, integration by parts is used to ``undo'' a product rule for derivatives:

$$ \frac{d(uv)}{dx} = u\,\frac{dv}{dx} + v \, \frac{du}{dx} \quad\implies\quad u\,\frac{dv}{dx} = \frac{d(uv)}{dx} - v \, \frac{du}{dx} $$

integrating both sides with respect to $x$:

$$ \int_{a}^{b}  u\,\frac{dv}{dx} \, dx = \int_{a}^{b} \left( \frac{d(uv)}{dx} - v \, \frac{du}{dx}  \right) \, dx $$

and using the fundamental theorem of calculus to simplify the first term on the right-hand side gives:

$$ \int_{a}^{b}  u\,\frac{dv}{dx} \, dx = \left.\frac{}{}uv\right|_{a}^{b} -  \int_{a}^{b} v \, \frac{du}{dx}  \, dx $$

In our case $t$ is our working variable, not $x$. Switching $x$ for $t$, we can let $\frac{dv}{dt} = \frac{df}{dt}$ so that $v(t) = f(t)$ and making $u(t) = e^{-st}$ so that $$ \frac{du}{dt} = -s e^{-st}  $$. This gives us the following transformation:

$$
\begin{aligned}
	\mathcal{L}\left\{\frac{d f}{dt}\right\}(s) &= \int_{0}^{\infty}  e^{-st} \,\frac{df}{dt} \, dt \\[1.15ex]
	&= \left.\frac{}{}e^{-st} f(t) \right|_{0}^{\infty} -  \int_{0}^{\infty} f(t) \, \left(-s e^{-st}\right)  \, dt \\[1.15ex]
	&= \left(  e^{-s(\infty)} f(\infty) - e^{-s(0)} f(0) \right) + s  \int_{0}^{\infty} f(t)  e^{-st}  \, dt \\[1.15ex]
	\mathcal{L}\left\{\frac{d f}{dt}\right\}(s) &= - f(0) + s  \int_{0}^{\infty} f(t)  e^{-st}  \, dt
\end{aligned}
$$

where we have assumed $e^{-st}$ decays to zero faster than $f(t)$ potentially diverges with an appropriate restriction to the domain of $s$. Also, notice the second term is now simply $s$ multiplied by the Laplace transform of $f(t)$. This gives us the following result : 

$$ \mathcal{L}\left\{\frac{d f}{dt}\right\}(s) = - f(0) + s \,  F(s) $$

Let's use an identical process, we can take the Laplace Transform of a second derivative to get:

$$ \mathcal{L}\left\{\frac{d^2 f}{dt^2}\right\}(s) = \int\limits_{0}^{\infty} e^{-s t} \frac{d^2 f}{dt^2} \, dt $$

We can let $\frac{dv}{dt} = \frac{d^2f}{dt^2}$ so that $v(t) = \frac{df}{dt} $ and we can also let $u(t) = e^{-st}$ so that 

$$ \frac{du}{dt} = -s e^{-st}  $$ 

This gives the following integration by parts:

$$
\begin{aligned}
	\mathcal{L}\left\{\frac{d^2 f}{dt^2}\right\}(s) &=  \int\limits_{0}^{\infty} e^{-s t} \frac{d^2 f}{dt^2} \, dt \\[1.15ex]
	&=  \left.\frac{}{}e^{-st} \frac{df}{dt} \right|_{0}^{\infty} -  \int_{0}^{\infty} \frac{df}{dt} \, \left(-s e^{-st}\right)  \, dt
\end{aligned}
$$

Again assuming, with an appropriate restitution to $s$, $e^{-st}$ goes to zero faster than $\frac{df(t)}{dt}$ potentially diverges, we can assume this term evaluated at $t = \infty$ is zero, and also notice the second term contains the Laplace transform of $\frac{df}{dt}$. We are left with

$$ \mathcal{L}\left\{\frac{d^2 f}{dt^2}\right\}(s) = -\left.\frac{df}{dt}\right|_0 + s \, \mathcal{L}\left[\frac{d f}{dt}\right](s) $$

Putting in what we found for the Laplace transform of a first derivative above, we get:

$$ \mathcal{L}\left\{\frac{d^2 f}{dt^2}\right\}(s) = -\left.\frac{df}{dt}\right|_0 - s f(0) + s^2 \,  F(s)$$

You can continue with higher and higher derivatives until you see a pattern that looks like:

$$ \mathcal{L}\left\{\frac{d^n f}{dt^n}\right\}(s) = s^n \,  F(s) - s^{n-1} f(0)  - s^{n-2} \left.\frac{df}{dt}\right|_0 \dots - s \left.\frac{d^{n-2}f}{dt^{n-2}}\right|_0 - \left.\frac{d^{n-1}f}{dt^{n-1}}\right|_0   $$

It is this that makes the Laplace Transformation so useful. Here are some observations we can make from this analysis of how derivatives transform:

- **Derivatives are turned into polynomials.**  
	Instead of dealing with derivatives, we use:

	$$
	\mathcal{L}\{f'(t)\} = sF(s) - f(0) \qquad \mathcal{L}\{f''(t)\} = s^2F(s) - sf(0) - f'(0)
	$$

	This means we replace $ f'' + 4f = 0 $ with something algebraic like $ s^2F(s) + 4F(s) = 0 $, which is way easier to solve.
	
- **They handle initial conditions naturally.**  
	Instead of solving for unknown constants separately, the **initial conditions are built into the transformed equation**.



{% capture ex %}
Suppose we want to solve:

$$
y'' + 4y = \sin(t), \quad y(0) = 0, \quad y'(0) = 1.
$$

Using direct methods, we would solve the homogeneous part, apply undetermined coefficients or variation of parameters, and handle the initial conditions carefully.

Instead we can apply the Laplace transform to get:

$$
\begin{aligned}
	\mathcal{L}\{y''\} + 4\mathcal{L}\{y\} &= \mathcal{L}\{\sin t\} \\[1.25ex]
	\Big( s^2Y(s) - sy(0) - y'(0) \Big) + 4\Big( sY(s) - y(0)\Big) &= \mathcal{L}\{\sin t\} \\[1.25ex]
	\Big( s^2Y(s) - 1 \Big) + 4\Big( sY(s) \Big) &= \mathcal{L}\{\sin t\} \\[1.25ex]
	s^2Y(s) + 4 s Y(s) - 1  &= \mathcal{L}\{\sin t\} \\[1.25ex]
\end{aligned}
$$

Now we just need to substitute in the Laplace Transform of $\sin(t)$, solve for $ Y(s) $ algebraically, and then inverse (undo) the transformation to get the solution to this ODE.

{% endcapture %}
{% include example.html content=ex %}




### Basic Properties of the Laplace Transform

To use the Laplace transform effectively, we need to understand some of its key properties. These properties allow us to manipulate the Laplace Transform in a fluid manner, making our work much more efficient.

#### Linearity

The Laplace transform is a linear operator, meaning that for any constants $ a $ and $ b $:

$$
\begin{aligned}
	\mathcal{L}\{a f(t) + b g(t)\} &= \int\limits_{0}^{\infty} e^{-s t} \Big( a f(t) + b g(t) \Big) \, dt \\[1.15ex]
	&= \int\limits_{0}^{\infty}  a e^{-s t} f(t) + b e^{-s t} g(t) \, dt \\[1.15ex]
	&= a \int\limits_{0}^{\infty}  e^{-s t} f(t) \, dt  + b \int\limits_{0}^{\infty}   e^{-s t} g(t) \, dt \\[1.15ex]
	\mathcal{L}\{a f(t) + b g(t)\} &= a \mathcal{L}\{f(t)\} + b \mathcal{L}\{g(t)\}
\end{aligned}
$$

This property allows us to break apart complex expressions and transform each term separately.

#### Translation (Shifting) in the $ \mathbf{s} $-Domain

If $ \mathcal{L}\{f(t)\} = F(s) $, then shifting $ f(t) $ by an exponential factor gives:

$$
\begin{aligned}
	\mathcal{L}\{e^{at} f(t)\} &= \int\limits_{0}^{\infty} e^{-s t} \Big( e^{at} f(t) \Big) \, dt \\[1.15ex]
	&= \int\limits_{0}^{\infty} \underbrace{e^{-(s-a) t} f(t)}_{\text{Let: } u = s - a} \, dt \\[1ex]
	&= \int\limits_{0}^{\infty} e^{-u t} f(t) \, dt \\[1.15ex]
	&= F(u)
\end{aligned}	
$$

and undo the substitution: $u = s - a$ to get:

$$ \mathcal{L}\{e^{at} f(t)\} = F(s - a) $$

This is particularly useful when solving ODEs with exponential terms in their solutions.

#### Laplace Transformation of Derivatives

As we have discussed already, one of the most useful properties of the Laplace Transform is how it handles derivatives. Instead of dealing with differentiation, we get simple algebraic expressions:

$$
\mathcal{L}\{f'(t)\} = sF(s) - f(0)
$$

$$
\mathcal{L}\{f''(t)\} = s^2 F(s) - s f(0) - f'(0)
$$

This is the reason Laplace Transforms are so effective for solving initial value problems: initial conditions appear naturally.

#### Laplace Transformation of Integrations

If we want to transform an integral, we can get the following:

$$
\begin{aligned}
	\mathcal{L} \left\{ \int_0^t f(\tau) \, d\tau \right\} &= \int\limits_{0}^{\infty} e^{-s t} \Big( \int_0^t f(\tau) \, d\tau \Big) \, dt \\[1.15ex]
	& \text{Let: } u = \int_0^t f(\tau) \, d\tau \quad\implies \frac{du}{dt} = f(t) \quad \text{and} \quad \frac{dv}{dt} = e^{-s t} \quad\implies v = -\frac{1}{s} e^{-s t} \\[1.15ex]
	&= - \Big(\int_0^t f(\tau) \, d\tau \Big) \, \Big(\frac{1}{s} e^{-s t}\Big) \Big|_0^\infty - \int\limits_{0}^{\infty} \Big(-\frac{1}{s} e^{-s t}\Big) \Big( f(t) \Big) \, dt      \\[1.15ex]
	&= - \Big(\underbrace{\int_0^0 f(\tau) \, d\tau}_{=0} \Big) \, \Big(\underbrace{\frac{1}{s} e^{-s (0)}}_{= 1/s}\Big) + \Big(\underbrace{\int_0^\infty f(\tau) \, d\tau}_{\neq \infty} \Big) \, \Big(\underbrace{\frac{1}{s} e^{-s (\infty)}}_{=0}\Big) + \frac{1}{s} \,\,\,\underbrace{\int\limits_{0}^{\infty} e^{-s t} f(t)  \, dt}_{\mathcal{L}\{f(t)\} = F(s)}      \\[1.15ex]
	&= 0 + 0 + \frac{1}{s} \, F(s) \\[1.15ex]
	\mathcal{L} \left\{ \int_0^t f(\tau) \, d\tau \right\} &= \frac{F(s)}{s} \\[1.15ex]
\end{aligned}
$$

This is useful for solving integral-differential equations. That is, differential equations with some terms that are integrals of the function we are looking for as opposed to ODEs with only derivatives of the sought after function.

#### Multiplication by $ \mathbf{t^n} $

Multiplying a function by a power of $ t $ in the time domain corresponds to differentiation in the $ s $-domain:

$$
\mathcal{L}\{t^n f(t)\} = (-1)^n \frac{d^n}{ds^n} F(s)
$$

This can be found by applying a cleaver derivative to the Laplace Transform of the function $f(t)$. For instance, what happens if we take a derivative with respect to $s$ of the transform of $f(t)$:

$$
\begin{aligned}
	\frac{d}{ds} \mathcal{L}\{f(t)\}(s) &= \frac{d}{ds} \, F(s) \\[1.15ex]
	&=  \frac{d}{ds} \Big(\int\limits_{0}^{\infty} e^{-s t} f(t) \, dt \Big) \\[1.15ex]
	&=  \int\limits_{0}^{\infty} \Big(\frac{de^{-s t}}{ds}\Big)  f(t) \, dt  \\[1.15ex]
	&=  \int\limits_{0}^{\infty} \Big(-te^{-s t}\Big)  f(t) \, dt  \\[1.15ex]
	&=  -\int\limits_{0}^{\infty} e^{-s t}  \,\, tf(t) \, dt  \\[1.15ex]
	\frac{d}{ds} \, F(s) &= - \mathcal{L}\{t \, f(t)\}(s)
\end{aligned}
$$

which gives us:

$$ \mathcal{L}\{t \, f(t)\}(s) = - \frac{d}{ds} \, F(s) = -\frac{d}{ds} \mathcal{L}\{f(t)\}(s) $$

We can take a second derivative with respect to $s$ to get:

$$ \mathcal{L}\{t^2 \, f(t)\}(s) = +\frac{d^2}{ds^2} \, F(s) = +\frac{d^2}{ds^2} \mathcal{L}\{f(t)\}(s) $$

which you are welcome to prove using the same method we applied for the first derivative. 

Repeating this for $n$ derivatives gives:

$$ \mathcal{L}\{t^n f(t)\} = (-1)^n \frac{d^n}{ds^n} F(s) $$











## Using Laplace Transforms to Solve ODEs

Now that we have built up an understanding of the Laplace transform and its properties, we can put it to work solving differential equations. The core idea is simple: the Laplace transform converts a differential equation into an algebraic equation, which is usually much easier to solve. Once we solve for $ F(s) $ in the $ s $-domain, we take the inverse Laplace transform (defined later) to return to the original function $ f(t) $.

### The General Strategy

The process for solving ODEs using the Laplace transform follows a structured approach:


- **Take the Laplace Transform of the ODE:** Apply $ \mathcal{L} $ to both sides of the equation, using known transforms and properties.
- **Solve for the transformed function, $ F(s) $:** The result will be an algebraic equation in $ s $, which we solve for $ F(s) $.
- **Take the Inverse Laplace Transform:** Use partial fractions if necessary and reference known inverse transforms to find $ f(t) $.
- **Interpret the Solution:** Check that the solution satisfies initial conditions and has the expected physical behavior.


The best way to see how this process plays out is to see it used in action. Let's consider a few examples to help build a solid understanding of this process. 


{% capture ex %}
Let's start with a simple first-order equation:

$$
\frac{dy}{dt} + 2y = 3e^{-t}
$$

with the initial condition, $y(0) = 1$. 

**Step 1: Take the Laplace Transform**

First let's apply the Laplace transform to both sides:

$$
\mathcal{L} \left\{ \frac{dy}{dt} \right\} + 2 \mathcal{L} \{ y \} = \mathcal{L} \{ 3e^{-t} \}
$$

Using the differentiation property and the transformation of an exponential we get:

$$
sY(s) - y(0) + 2Y(s) = \frac{3}{s+1}
$$

Since $ y(0) = 1 $, we substitute that in to get:

$$
sY(s) - 1 + 2Y(s) = \frac{3}{s+1}
$$

**Step 2: Solve for $ Y(s) $**

Algebraically rearranging the equation to solve for $Y(s)$:

$$
\begin{aligned}
	sY(s) - 1 + 2Y(s) &= \frac{3}{s+1} \\[1.15ex]
	(s + 2 ) Y(s) &= \frac{3}{s+1} + 1 \\[1.15ex]
	Y(s) &= \frac{3}{(s+1)(s+2)} + \frac{1}{s+2}
\end{aligned}
$$


**Step 3: Take the Inverse Laplace Transform** 

Before taking the inverse transformation it is typically helpful to use partial fraction decomposition to make the inverse transform easier to compute. This is not always going to be possible, but when it is it is immensely helpful. In this case we can use partial fraction decomposition on the first term on the left hand side:

$$ \frac{3}{(s+1)(s+2)} = \frac{A}{s + 1} + \frac{B}{s + 2} \qquad \implies \qquad 3 = A(s+2) + B (s + 1) $$

Taking $s = -1$ gives $3 = A(1) + 0 \implies A = 3$. Similarly, taking $s = -2$ gives $3 = 0 + B(-1) \implies B = -3$. This leaves us with:

$$ \frac{3}{(s+1)(s+2)} = \frac{3}{s + 1} - \frac{3}{s + 2} $$

and the right hand side of the transformed ODE simplifies to:

$$
\begin{aligned}
	Y(s) &= \frac{3}{(s+1)(s+2)} + \frac{1}{s+2} \\[1.15ex]
	&= \frac{3}{s + 1} - \frac{3}{s + 2} + \frac{1}{s+2} \\[1.15ex]
	Y(s) &= \frac{3}{s + 1} - \frac{2}{s + 2} \\[1.15ex]
\end{aligned}
$$

Now we can take the inverse Laplace Transform of both sides: 

$$ \mathcal{L}^{-1}\Big\{ Y(s) \Big\}  = \mathcal{L}^{-1}\Big\{ \frac{3}{s + 1} \Big\}  - \mathcal{L}^{-1}\Big\{ \frac{2}{s + 2} \Big\} $$

Looking the terms in the right hand side up in the table given above we can take the following inverse transformations:

$$ \mathcal{L}^{-1}\Big\{ \frac{3}{s + 1} \Big\} = \mathcal{L}^{-1}\Big\{ \frac{3}{s - (-1) } \Big\} = 3e^{-t}  \qquad\text{and}\qquad \mathcal{L}^{-1}\Big\{ \frac{2}{s + 2} \Big\} =  \mathcal{L}^{-1}\Big\{ \frac{2}{s - (-2)} \Big\} = 2e^{-2t}$$

Substituting these in gives us the solution:

$$
y(t) = 3e^{-t} - 2e^{-2t}
$$

**Step 4: Interpret the Solution**

A quick check to make sure the initial conditions are correct:

$$
y(0) = 3e^{0} - 2e^{0} = 3 - 2 = 1
$$

and the steady state solution (derivatives go to zero and the limit of $t$ goes to infinity) lines up:

$$ \frac{dy}{dt} + 2y = 3e^{-t} \qquad\implies\qquad 0 + 2y_\text{Steady} = 3e^{-(\infty)} \qquad\implies\qquad y_\text{Steady} = 0 $$

Comparing this to out solution with $t$ taken to imfinity:

$$ y(t) = 3e^{-t} - 2e^{-2t} \qquad\implies\qquad y(t\rightarrow\infty) = 3(0) - 2(0) = 0 $$

Everything appears to be in good order! 
{% endcapture %}
{% include example.html content=ex %}




{% capture ex %}
Now let’s apply the Laplace transform to a second-order equation:

$$
y'' + 3y' + 2y = e^{-t}
$$

with the initial conditions: $y(0) = 0$ and  $y'(0) = 1$. 

**Step 1: Take the Laplace Transform**

Let's apply the Laplace transform to both sides:

$$
\mathcal{L} \{ y'' \} + 3 \mathcal{L} \{ y' \} + 2\mathcal{L} \{ y \} =  \mathcal{L} \{ e^{-t} \}
$$

Using the differentiation property and the transformation of an exponential we get:

$$
\Big(s^2 Y(s) - sy(0) - y'(0) \Big) + 3\Big(sY(s) - y(0)\Big) + 2Y(s) = \frac{1}{s+1}
$$

and applying the initial conditions:

$$
s^2 Y(s) - 1 + 3sY(s) + 2Y(s) = \frac{1}{s+1}
$$

**Step 2: Solve for $ Y(s) $**

Factoring out $ Y(s) $:

$$
(s^2 + 3s + 2) Y(s) = \frac{1}{s+1} + 1
$$

and dividing by $ s^2 + 3s + 2 = (s+1)(s+2) $, we rewrite:

$$
Y(s) = \frac{1}{(s+1)^2(s+2)} + \frac{1}{(s+1)(s+2)}
$$

In this case, combining the right hand side into one fraction will be easier to handle since applying partial fraction decomposition once is preferable to having to apply it twice:

$$
Y(s) = \frac{1}{(s+1)^2(s+2)} + \frac{s + 1}{(s+1)^2(s+2)} \quad\implies\quad Y(s) = \frac{s + 2}{(s+1)^2(s+2)}
$$

Canceling out the $ (s+2) $ in the numerator and denominator leaves us with:

$$
Y(s) = \frac{1}{(s+1)^2}
$$

**Step 3: Take the Inverse Laplace Transform**

In this case partial fractions is not necessary. Notice the right-hand side of the previous equation can be found in the Laplace Transform table using:

$$ \mathcal{L}\Big\{t^n e^{bt}\Big\} = \frac{n!}{(s - b)^{n+1}} \qquad ( s > b ) $$

To match this to what we have on the right-hand side of the equation for $Y(z)$ we can take $n = 1$ and $ b = -1$:

$$ \mathcal{L}\Big\{t^1 e^{(-1)t}\Big\} = \frac{1!}{(s - (-1))^{1+1}} = \frac{1}{(s + 1)^{2}}  \qquad ( s > -1 ) $$

Applying the inverse transform thus gives us:

$$
y(t) = t e^{-t}
$$

**Step 4: Interpret the Solution** 

We can check this answer by first considering the initial conditions:

$$ y(t) = t e^{-t} \qquad\implies\qquad  y(0) = 0 e^{-0} = 0 $$

$$ y'(t) = e^{-t} - t e^{-t}  \qquad\implies\qquad  y'(0) =  e^{-0} - 0 e^{-0} = 1 + 0 = 1 $$

Secondly we can check the steady state behavior:

$$ y'' + 3y' + 2y = e^{-t}  \qquad\implies\qquad 0 + 3(0) + 2y_\text{Steady} = e^{-(\infty)}  \qquad\implies\qquad y_\text{Steady} = 0 $$

and taking the limit as $t$ goes to infinity for our solution:

$$ y(t) = t e^{-t}  \qquad\implies\qquad y(t\rightarrow \infty) = (\infty) e^{-(\infty)} = 0 $$

where this limit goes to zero from L'Hospital's Rule.

Everything is consistent, which suggests our solution is correct within these limits. 
{% endcapture %}
{% include example.html content=ex %}












## Problems:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.




**Problem 1: Solving an Initial Value Problem**

Solve the following initial value problem using Laplace Transforms:

$$
y'' + 4y' + 5y = 0, \quad y(0) = 2, \quad y'(0) = -1.
$$

Express your final answer in terms of exponentials and sines/cosines.

**Problem 2: Forced Oscillations and Laplace Transforms**

A damped oscillator is governed by:

$$
m \frac{d^2x}{dt^2} + b \frac{dx}{dt} + kx = F_0 e^{-t}, \quad x(0) = 0, \quad x'(0) = 0.
$$

Using Laplace Transforms, solve for $ x(t) $. Assume $ m = 1 $, $ b = 2 $, and $ k = 5 $.

















