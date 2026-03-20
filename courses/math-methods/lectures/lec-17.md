---
layout: default
title: Mathematical Methods - Lecture 17
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 17
---

# Lecture 17 – Laplace Transformations I



## Introduction: The What and Why of Laplace Transforms

Up until now, we’ve tackled differential equations using different direct solution techniques such as the methods of separation of variables, integrating factors, characteristics (guess an exponential), undetermined coefficients (guess and check), and variation of parameters, to name a few. However, sometimes solving an ODE directly isn’t the easiest approach. 

Imagine how much easier it would be is we could transform the differential equation into an algebraic equation. We could solve it as it if were a regular algebra problem and could avoid wrestling with all the derivatives, integrals, and multiple solution steps. 

But what kind of transformation could do this for us? 

It turns out there are a multitude of options, but the most popular options by far are the Laplace and Fourier Transformations.

**Laplace and Fourier Transformations** convert differential equations into algebraic problems, making solving a differential equation significantly easier. In this course we will discuss the Laplace Transformation, and the Fourier Transformation will be handled in Mathematical Physics when we have more familiarity with functional orthogonality and other related topics.





### The What: Laplace Transformation

The **Laplace transform** takes a function $ f(t) $ whose working variable is $t$ (depending on what $t$ represents this could be time, space, etc.), and transforms it into a function $ F(s) $, a function of $s$, the Laplace domain (which could be frequency, wave-number, etc. depending on what $t$ represents). To transform from $f(t)$ to $F(s)$, that is the Laplace transformation of $f(t)$ is given as:

$$
\mathcal{L}\{f(t)\}(s) = F(s) = \int_0^\infty e^{-st} f(t) \, dt
$$

Notice the working variable $t$ is integrated over a set of definite limits. This means there will be no $t$'s in the result of this integration. This is why we say this transformations shifts us from the $t$ domain (or representation) to the $s$ domain (or representation). 


In general $ s $ can be a complex variable, but we will take it to be strictly real for this discussion. Something we will need to pay strict attention to is ensuring this integration actually converges for whichever function $f(t)$ is being transformed. This can typically be done by restricting the domain of $s$ (i.e. the range of values $s$ can take) to ensure the integration of the Laplace Transformation will converge. Let's consider an example of how the domain of $s$ can be restricted to ensure convergence.

Let’s calculate the Laplace transform of a simple exponential function, $f(t) = e^{bt}$. By definition, this is given by the integral

$$
\mathcal{L}\{f(t)\}(s) = F(s) = \int_{0}^{\infty} e^{-st} e^{bt} \, dt = \int_{0}^{\infty} e^{-(s-b)t} \, dt
$$

For this integral to converge, the exponential must decay to zero as $t \to \infty$. This requires

$$
-(s - b) < 0 \quad \implies \quad s - b > 0 \quad \implies \quad s > b
$$

So, the transform $F(s)$ is only defined for values of $s$ greater than $b$. This condition is often referred to as the region of convergence.

It is fairly easy to see why $s = b$ does not work. If $s = b$, then the exponent becomes zero and the integrand is simply $1$. In that case, we are integrating a constant over an infinite interval, which diverges.

What about values where $s < b$? In that case, the exponent becomes positive, so the exponential grows as $t$ increases. This causes the integral to diverge as well.

All of this ties back to the limits of integration in the Laplace transform. Since we are integrating from $0$ to $\infty$, we need the integrand to decay as $t$ becomes large. That requirement is what enforces the condition $s > b$.

If the limits of integration were different, the condition on $s$ would change as well. For example, if we were integrating from $-\infty$ to $0$, convergence would instead require $s < b$.

It is important to note that the limits of the Laplace transform are not necessarily the same for all situations. The standard definition we have been using:

$$
\mathcal{L}\{f(t)\} = \int_{0}^{\infty} e^{-st} f(t)\,dt
$$

is often referred to as the one-sided Laplace transform. This form is particularly useful in physics and engineering because it models systems that begin at $t = 0$, such as turning on a circuit or releasing a mass from rest.

However, there is also a two-sided Laplace transform, defined as

$$
\mathcal{L}\{f(t)\} = \int_{-\infty}^{\infty} e^{-st} f(t)\,dt.
$$

In this case, the conditions for convergence are different, since the integrand must behave well as $t \to \infty$ *and* as $t \to -\infty$.

This highlights an important point: the limits of integration directly affect the conditions required for convergence. In the one-sided transform, we only need the function to behave well as $t \to \infty$, which leads to conditions like $s > b$ in our example. In the two-sided case, we must consider behavior in both directions, which can further restrict the allowable values of $s$.

The key takeaway is that the definition of the transform and its limits determine where the transform exists. It is always worth checking that the integral converges under the limits being used.

Now, picking up where we left off, we can complete the integration to get:

$$
\begin{aligned}
F(s)&= \mathcal{L}\{f(t)\}(s) \\[1.15ex]
	&= \int_{0}^{\infty} e^{-(s-b) t} \, dt \\[1.15ex]
	&= \frac{-1}{s -b} \left.\left( e^{-(s-b)t}\right)\right|_{0}^{\infty} \, dt \\[1.15ex]
	&= \frac{-1}{s -b} \left( 0 - e^{-(s-b)0}\right) \, dt \\[1.15ex]
	&= \frac{-1}{s -b} \left( - 1\right) \\[1.15ex]
F(s)&= \frac{1}{s -b}
\end{aligned}
$$


So, the Laplace Transformation of an exponential $e^{bt}$ is this fraction, where $s > b$. This means, for example, if $b=5$ then:

$$ f(t) = e^{5t} \qquad \longrightarrow \qquad  F(s) = \frac{1}{s -5}$$

where $s$ can only take on values larger than $5$. 



### Table of Common Laplace Transforms

To streamline calculations, we often use a reference table of common transforms. Here are a few that will frequently appear:

| Given $f(t)$ | Transforms to $F(s) = \mathcal{L}\{f(t)\}$ |
|---|---|
| $\frac{d}{dt} f(t)$ | $sF(s) - f(0)$ |
| $\frac{d^2}{dt^2} f(t)$ | $s^2 F(s) - s f(0) - f'(0)$ |
|  |  |
| $1$ | $\frac{1}{s}$ &emsp; ($s > 0$) |
| $t^n$ | $\frac{n!}{s^{n+1}}$ &emsp; ($s > 0$) |
| $e^{bt}$ | $\frac{1}{s - b}$ &emsp; ($s > b$) |
| $t^n e^{bt}$ | $\frac{n!}{(s - b)^{n+1}}$ &emsp; ($s > b$) |
| $\frac{1}{(s - b)^n}$ | $\frac{t^{n-1} e^{bt}}{(n-1)!}$ &emsp; ($n \geq 1$) |
|  |  |
| $\cos(bt)$ | $\frac{s}{s^2 + b^2}$ &emsp; ($s > 0$) |
| $\sin(bt)$ | $\frac{b}{s^2 + b^2}$ &emsp; ($s > 0$) |
| $\sinh(bt)$ | $\frac{b}{s^2 - b^2}$ &emsp; ($s > \vert b\vert $) |
| $\cosh(bt)$ | $\frac{s}{s^2 - b^2}$ &emsp; ($s > \vert b\vert $) |
| $e^{-at}\sin(bt)$ | $\frac{b}{(s + a)^2 + b^2}$ &emsp; ($s > -a$) |
| $e^{-at}\cos(bt)$ | $\frac{s + a}{(s + a)^2 + b^2}$ &emsp; ($s > -a$) |
| $\frac{\sin(bt)}{t}$ | $\tan^{-1}(b/s)$ &emsp; ($s > 0$) |
|  |  |
| $\frac{1}{\sqrt{t}}$ | $\sqrt{\frac{\pi}{s}}$ &emsp; ($s > 0$) |
| $\frac{t^m}{\Gamma(m+1)}$ | $\frac{1}{s^{m+1}}$ &emsp; ($s > 0$, $m > -1$) |
| $\frac{1 - e^{-at}}{t}$ | $\ln\left(\frac{s+a}{s}\right)$ &emsp; ($s > -a$) |
| $\frac{e^{-at}}{t}$ | $-\text{Ei}(-as)$ |
|  |  |
| $U(t-c)$ (unit step at $t=c$) | $\frac{e^{-cs}}{s}$ |
| $U(t-c)f(t-c)$ (shifted function) | $e^{-cs}F(s)$ |
| $t\,U(t-c)$ (ramp function) | $\frac{e^{-cs}}{s^2}$ |
| $\delta(t-c)$ (Dirac delta) | $e^{-cs}$ |







### The Why: Changing ODEs to Algebra Problems

We made a bold claim in the introduction to this Lecture: the Laplace Transformation can change solving an ODE to solving an algebra problem. Let's see how this can be done.

A special property of the Laplace transformation is how it interacts with derivatives. This is difficult to put into words, to let's  jump into calculating the Laplace transformation of the first derivative of a function to see what happens:

$$ \mathcal{L}\left\{\frac{d f}{dt}\right\}(s) = \int_{0}^{\infty} e^{-s t} \frac{d f}{dt} \, dt $$

We can strip that derivative off of the function $f(t)$ using integration by parts. Recall, integration by parts is used to "undo" a product rule for derivatives:

$$ \frac{d(uv)}{dx} = u\,\frac{dv}{dx} + v \, \frac{du}{dx} \quad\implies\quad u\,\frac{dv}{dx} = \frac{d(uv)}{dx} - v \, \frac{du}{dx} $$

integrating both sides with respect to $x$:

$$ \int_{a}^{b}  u\,\frac{dv}{dx} \, dx = \int_{a}^{b} \left( \frac{d(uv)}{dx} - v \, \frac{du}{dx}  \right) \, dx $$

and using the fundamental theorem of calculus to simplify the first term on the right-hand side gives:

$$ \int_{a}^{b}  u\,\frac{dv}{dx} \, dx = \left.\frac{}{}uv\right|_{a}^{b} -  \int_{a}^{b} v \, \frac{du}{dx}  \, dx $$

In our case $t$ is our working variable, not $x$. Switching $x$ for $t$, we can let $\frac{dv}{dt} = \frac{df}{dt}$ so that $v(t) = f(t)$ and making $u(t) = e^{-st}$ so that $ \frac{du}{dt} = -s e^{-st}  $. This gives us the following transformation:

$$
\begin{aligned}
	\mathcal{L}\left\{\frac{d f}{dt}\right\}(s) &= \int_{0}^{\infty}  e^{-st} \,\frac{df}{dt} \, dt \\[1.15ex]
	&= \left.\frac{}{}e^{-st} f(t) \right|_{0}^{\infty} -  \int_{0}^{\infty} f(t) \, \left(-s e^{-st}\right)  \, dt \\[1.15ex]
	&= \left(  e^{-s(\infty)} f(\infty) - e^{-s(0)} f(0) \right) + s  \int_{0}^{\infty} f(t)  e^{-st}  \, dt \\[1.15ex]
	&= \left(  0 - f(0) \right) + s  \int_{0}^{\infty} f(t)  e^{-st}  \, dt \\[1.15ex]
	\mathcal{L}\left\{\frac{d f}{dt}\right\}(s) &= - f(0) + s  \int_{0}^{\infty} f(t)  e^{-st}  \, dt
\end{aligned}
$$

where we have assumed $e^{-st}$ decays to zero faster than $f(t)$ potentially diverges as $t\rightarrow \infty$ with any appropriate restrictions to the domain of $s$ included. Also, notice the second term is now simply $s$ multiplied by the Laplace transform of $f(t)$. This gives us the following result: 

$$ \mathcal{L}\left\{\frac{d f}{dt}\right\}(s) = - f(0) + s \,  F(s) $$

Let's use an identical process to see what happens when we take the Laplace Transform of a second derivative:

$$ \mathcal{L}\left\{\frac{d^2 f}{dt^2}\right\}(s) = \int_{0}^{\infty} e^{-s t} \frac{d^2 f}{dt^2} \, dt $$

We apply integration by parts again, letting $\frac{dv}{dt} = \frac{d^2f}{dt^2}$ so that $v(t) = \frac{df}{dt} $ and $u(t) = e^{-st}$ so that $ \frac{du}{dt} = -s e^{-st}  $ to get:

$$
\begin{aligned}
	\mathcal{L}\left\{\frac{d^2 f}{dt^2}\right\}(s) &=  \int_{0}^{\infty} e^{-s t} \frac{d^2 f}{dt^2} \, dt \\[1.15ex]
	&=  \left.\frac{}{}e^{-st} \frac{df}{dt} \right|_{0}^{\infty} -  \int_{0}^{\infty} \frac{df}{dt} \, \left(-s e^{-st}\right)  \, dt
\end{aligned}
$$

Again assuming, with appropriate restrictions to $s$, $e^{-st}$ goes to zero faster than $\frac{df(t)}{dt}$ potentially diverges as $t \rightarrow \infty$, we can assume this limit in the first term is zero. Also, notice the second term contains the Laplace transform of $\frac{df}{dt}$. We are left with

$$ \mathcal{L}\left\{\frac{d^2 f}{dt^2}\right\}(s) = -\left.\frac{df}{dt}\right|_0 + s \, \mathcal{L}\left\{\frac{d f}{dt}\right\}(s) $$

Putting in what we found for the Laplace transform of a first derivative above, we get:

$$ \mathcal{L}\left\{\frac{d^2 f}{dt^2}\right\}(s) = -\left.\frac{df}{dt}\right|_0 - s f(0) + s^2 \,  F(s)$$

You can continue with higher and higher derivatives until you see a pattern that looks like:

$$ \mathcal{L}\left\{\frac{d^n f}{dt^n}\right\}(s) = s^n \,  F(s) - s^{n-1} f(0)  - s^{n-2} \left.\frac{df}{dt}\right|_0 \dots - s \left.\frac{d^{n-2}f}{dt^{n-2}}\right|_0 - \left.\frac{d^{n-1}f}{dt^{n-1}}\right|_0   $$

It is this that makes the Laplace Transformation so useful: It transforms derivatives into polynomials! 

Here is a list of interesting and important observations we can make from this analysis of how derivatives transform:

- **Derivatives are turned into polynomials.**  
	Instead of dealing with derivatives, we use:

	$$
	\mathcal{L}\{f'(t)\} = sF(s) - f(0) \qquad \mathcal{L}\{f''(t)\} = s^2F(s) - sf(0) - f'(0)
	$$

	This means we replace $ f'' + 4f = 0 $ will transform into something algebraic like $ s^2F(s) + 4F(s) = 0 $, which is way easier to solve.
	
- **They handle initial conditions naturally.**  
	Instead of solving for unknown constants separately, the **initial conditions are built into the transformed equation**.

Let's put this tool into action by looking at an example.

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

Now we just need to substitute in the Laplace Transform of $\sin(t)$, solve for $ Y(s) $ algebraically, and then inverse (undo) the transformation to get the solution to this ODE. We will see this process in full in a few examples at the end of this lecture. 

{% endcapture %}
{% include example.html content=ex %}




### Basic Properties of the Laplace Transform

To use the Laplace transform effectively, we need to understand some of its key properties. These properties allow us to manipulate the Laplace Transform in a fluid manner, making our work much more efficient.

#### Linearity

You may not have noticed, but we actually used this property in the previous example! The Laplace transform is a linear operator, meaning that for any constants $ a $ and $ b $ and functions $f(t)$ and $g(t)$:

$$
\begin{aligned}
	\mathcal{L}\{a f(t) + b g(t)\} &= \int_{0}^{\infty} e^{-s t} \Big( a f(t) + b g(t) \Big) \, dt \\[1.15ex]
	&= \int_{0}^{\infty}  a e^{-s t} f(t) + b e^{-s t} g(t) \, dt \\[1.15ex]
	&= a \int_{0}^{\infty}  e^{-s t} f(t) \, dt  + b \int_{0}^{\infty}   e^{-s t} g(t) \, dt \\[1.15ex]
	\mathcal{L}\{a f(t) + b g(t)\} &= a \mathcal{L}\{f(t)\} + b \mathcal{L}\{g(t)\}
\end{aligned}
$$

This property allows us to break apart complex expressions and transform each term separately.

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
	\mathcal{L} \left\{ \int_0^t f(\tau) \, d\tau \right\} &= \int_{0}^{\infty} e^{-s t} \Big( \int_0^t f(\tau) \, d\tau \Big) \, dt \\[1.15ex]
	& \text{Let: } u = \int_0^t f(\tau) \, d\tau \implies \frac{du}{dt} = f(t) \quad \text{and} \quad \frac{dv}{dt} = e^{-s t} \implies v = -\frac{1}{s} e^{-s t} \\[1.15ex]
	&= - \Big(\int_0^t f(\tau) \, d\tau \Big) \, \Big(\frac{1}{s} e^{-s t}\Big) \Big|_0^\infty - \int_{0}^{\infty} \Big(-\frac{1}{s} e^{-s t}\Big) \Big( f(t) \Big) \, dt      \\[1.15ex]
	&= - \Big(\int_0^0 f(\tau) \, d\tau \Big) \, \Big(\frac{1}{s} e^{-s (0)}\Big) + \Big(\int_0^\infty f(\tau) \, d\tau \Big) \, \Big(\frac{1}{s} e^{-s (\infty)}\Big) + \frac{1}{s} \,\,\,\int_{0}^{\infty} e^{-s t} f(t)  \, dt
\end{aligned}
$$
Inspecting this line of math we can make a couple of important observations:

- The integral in the first term will be zero since the limits of the integral are the same. 
- The exponential in the first term will go to 1, leaving that parenthesis as $\frac{1}{s}$. 
	- Combining these results tells us that the first term will vanish entirely. 
- We will need to assume the integral of $f(\tau)$ in the second term either does not diverge, or, if it does, it diverges slower than the exponential in neighboring parenthesis converges to zero.
	- This restricts the kind of function $f$ can be and also forces us to impose the restriction that $s > 0 $ so that the exponential will converge to zero. 

Putting all of this together leaves us with the following result: 


$$
\begin{aligned}
	\mathcal{L} \left\{ \int_0^t f(\tau) \, d\tau \right\} &= 0 + 0 + \frac{1}{s} \, F(s) \\[1.15ex]
	&= \frac{F(s)}{s} \\[1.15ex]
\end{aligned}
$$

This result is useful for solving integral-differential equations. That is, differential equations with some terms that are integrals of the function we are looking for as opposed to ODEs with only derivatives of the sought after function.

#### Translation (Shifting) in the $ \mathbf{s} $-Domain

If $ \mathcal{L}\{f(t)\} = F(s) $, then multiplying $ f(t) $ by an exponential factor like $e^{at} f(t)$ transforms as:

$$
\begin{aligned}
	\mathcal{L}\{e^{at} f(t)\} &= \int_{0}^{\infty} e^{-s t} \Big( e^{at} f(t) \Big) \, dt \\[1.15ex]
	&= \int_{0}^{\infty} e^{-(s-a) t} f(t) \, dt
\end{aligned}	
$$

Here we can simplify this integral by making the following $u$-substitution, $u = s - a$, to get:

$$
\begin{aligned}
	\mathcal{L}\{e^{at} f(t)\} &= \int_{0}^{\infty} e^{-u t} f(t) \, dt \\[1.15ex]
	&= F(u)
\end{aligned}	
$$

where the above integration is just the Laplace Transformation but using $u$ as the Laplace domain variable instead of $s$. Undoing the substitution we have:

$$ \mathcal{L}\{e^{at} f(t)\} = F(s - a) $$

This will give us the same function as transforming $f(t)$, however the result will be translated to be centered on $ s=a $ in the Laplace space. 

This is particularly useful when solving ODEs with exponential terms in their solutions.


#### Multiplication by $ \mathbf{t^n} $

Multiplying a function by a power of $ t $ in the time domain corresponds to differentiation in the $ s $-domain:

$$
\mathcal{L}\{t^n f(t)\} = (-1)^n \frac{d^n}{ds^n} F(s)
$$

This can be found by applying a cleaver derivative to the Laplace Transform of the function $f(t)$. For instance, what happens if we take a derivative with respect to $s$ of the transform of $f(t)$:

$$
\begin{aligned}
	\frac{d}{ds} \mathcal{L}\{f(t)\}(s) &= \frac{d}{ds} \, F(s) \\[1.15ex]
	&=  \frac{d}{ds} \Big(\int_{0}^{\infty} e^{-s t} f(t) \, dt \Big) \\[1.15ex]
	&=  \int_{0}^{\infty} \Big(\frac{de^{-s t}}{ds}\Big)  f(t) \, dt  \\[1.15ex]
	&=  \int_{0}^{\infty} \Big(-te^{-s t}\Big)  f(t) \, dt  \\[1.15ex]
	&=  -\int_{0}^{\infty} e^{-s t}  \,\, tf(t) \, dt  \\[1.15ex]
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

Now that we have built up an understanding of the Laplace transform and its properties, we can put it to work solving differential equations. The core idea is simple: the Laplace transform converts a differential equation into an algebraic equation, which is usually much easier to solve. Once we solve for $ F(s) $ in the $ s $-domain, we take the inverse Laplace transform to return to the original function $ f(t) $.

### The General Strategy

The process for solving ODEs using the Laplace transform follows a structured approach:


- **Take the Laplace Transform of the ODE:** Apply $ \mathcal{L} $ to both sides of the equation, using known transforms and properties.
- **Solve for the transformed function, $ F(s) $:** The result will be an algebraic equation in $ s $, which we solve for $ F(s) $.
- **Take the Inverse Laplace Transform:** Use partial fractions, if necessary, and reference known inverse transforms to find $ f(t) $.
	- You should not have to do any integrations here. The key is to mold the result into a form that matches on of the functions in the Laplace Transformation table given at the beginning of this lecture. This will make more sense when we do some examples, so we will refrain from attempting to explain how this is done here. 
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
sY(s) - y(0) + 2Y(s) = \frac{3}{s - (-1)}
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

$$ \frac{3}{(s+1)(s+2)} = \frac{A}{s + 1} + \frac{B}{s + 2} \quad \implies \quad 3 = A(s+2) + B (s + 1) $$

Taking $s = -1$ gives $3 = A(1) + 0 \implies A = 3$. 

Similarly, taking $s = -2$ gives $3 = 0 + B(-1) \implies B = -3$. 

This leaves us with:

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

Comparing the terms in the right hand side to the results in the Laplace Transformation table given at the beginning of this lecture we can find the following results:

$$ \mathcal{L}^{-1}\Big\{ \frac{3}{s + 1} \Big\} = \mathcal{L}^{-1}\Big\{ \frac{3}{s - (-1) } \Big\} = 3e^{-t}$$

and

$$\mathcal{L}^{-1}\Big\{ \frac{2}{s + 2} \Big\} =  \mathcal{L}^{-1}\Big\{ \frac{2}{s - (-2)} \Big\} = 2e^{-2t}$$

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

Comparing this to our solution with $t$ taken to infinity:

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
\Big(s^2 Y(s) - sy(0) - y'(0) \Big) + 3\Big(sY(s) - y(0)\Big) + 2Y(s) = \frac{1}{s - (-1)}
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

Let's combining the right hand side into one fraction will be easier to handle since applying partial fraction decomposition once is preferable to having to apply it twice:

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
y'' + 4y' + 5y = 0, \quad y(0) = 2, \quad y'(0) = -1
$$

Express your final answer in terms of exponentials and sines/cosines.

**Problem 2: Forced Oscillations and Laplace Transforms**

A damped oscillator is governed by:

$$
m \frac{d^2x}{dt^2} + b \frac{dx}{dt} + kx = F_0 e^{-t}, \quad x(0) = 0, \quad x'(0) = 0
$$

Using Laplace Transforms, solve for $ x(t) $. Assume $ m = 1 $, $ b = 2 $, and $ k = 5 $.

















