---
layout: default
title: Pysics for Life Sciences I - Lecture 02
course_home: /courses/phys-for-life-sci-i/
nav_section: lectures
nav_order: 2
---

# Lecture 02 — Velocity


## Review: Trigonometry and Adding Vectors (from Last Lecture)

In the previous lecture, we introduced how trigonometry is used in physics to work with vectors. The main idea is that right triangles allow us to move back and forth between:

- A single vector described by a **magnitude and direction**
- The same vector described by **horizontal and vertical components**

### Right-Triangle Definitions

For a right triangle with a given angle $\theta$:

- The **hypotenuse** is the side opposite the right angle  
- The **adjacent** side is next to $\theta$, but not the hypotenuse  
- The **opposite** side is across from $\theta$  

Which side is adjacent or opposite depends entirely on the angle you choose to use as your reference.

### Combining Components into a Magnitude and Direction

If the horizontal and vertical components are known:

- The **magnitude** of the vector is found using the Pythagorean Theorem
  - When taking square roots, both positive and negative answers appear mathematically. It is your job to choose the answer that makes physical sense.
- The **direction** of the vector is found using inverse tangent

An angle by itself does not fully describe a direction. A proper direction must include:

- A **reference direction** (such as East or North)
- A **direction to turn** from that reference  

We write directions in the form:

$$
\text{Angle; Turn-of-Reference}
$$

For example, “$37^\circ$ North of East” means:
- Start by facing East
- Turn $37^\circ$ toward North

This language will be used consistently whenever vector directions are reported.

### Big Picture Takeaways

- Trigonometry is the tool that connects the components of a vector to its magnitude and direction  
- Components are found using sine and cosine  
- Magnitude and direction are recovered using the Pythagorean Theorem and tangent  
  - Direction must always include a turn-of-reference  

Let's reinforce this, and review adding vectors together, by considering a slightly more difficult vector addition example compared to the ones we considered in Lecutre 01.

{% capture ex %}
In this problem, an ultimate Frisbee player runs a pattern made up of three displacement vectors, $\vec{A}$, $\vec{B}$, and $\vec{C}$. We want the **resultant displacement**, meaning the single displacement vector $\vec{R}$ that takes the player from the starting point directly to the ending point.

The given information is:

- $\vec{A} = 10\ \text{m}$ (straight up)
- $\vec{B} = 30\ \text{m}$ (to the right)
- $\vec{C} = 34\ \text{m}$ at an angle of $37^\circ$ below the positive $x$ direction  
  (so it points down and to the right)

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec02/UltimateFrisbee.png' | relative_url }}"
  alt="The diagram shows three displacement vectors arranged head-to-tail. Vector A is a vertical arrow pointing upward with a magnitude of 10 meters. From the tip of vector A, vector B is drawn horizontally to the right with a magnitude of 30 meters. From the tip of vector B, vector C is drawn down and to the right at an angle of 37 degrees below the horizontal, with a magnitude of 34 meters. A dashed arrow labeled R runs from the starting point of vector A to the ending point of vector C, representing the resultant displacement for the entire motion."
  style="display:block; margin:1.5rem auto; max-width:800px; width:60%;">

We want:
1. The magnitude of $\vec{R}$
2. The direction of $\vec{R}$ using turn-of-reference language


### Step 1: Break Each Vector into Components

We will treat right as $+x$ and up as $+y$.

#### Vector $\vec{A}$

$\vec{A}$ points up with magnitude $10\ \text{m}$.

- $A_x = 0\ \text{m}$
- $A_y = +10\ \text{m}$

#### Vector $\vec{B}$

$\vec{B}$ points to the right with magnitude $30\ \text{m}$.

So its components are:

- $B_x = +30\ \text{m}$
- $B_y = 0\ \text{m}$

#### Vector $\vec{C}$

$\vec{C}$ has magnitude $34\ \text{m}$ and is $37^\circ$ below the positive $x$ direction (South of East).

That means:
- The horizontal component is positive (to the right)
- The vertical component is negative (down)

Using soem trigonomentry, we have:

$$
C_x = (34\ \text{m})\cos(37^\circ) = (34)\cos(37^\circ) = 27.1\ \text{m}
$$

$$
C_y = -(34\ \text{m})\sin(37^\circ) = -(34)\sin(37^\circ) = -20.5\ \text{m}
$$


### Step 2: Add Components to Get the Resultant Components

Putting these resulting into our typical organizational table gives:

| Vector | Right | Up |
|:------:|:-----------:|:------------:|
| $\vec{A}$ | $0$ | $+10\ \text{m}$ |
| $\vec{B}$ | $+30\ \text{m}$ | $0$ |
| $\vec{C}$ | $+27.1\ \text{m}$ | $-20.5\ \text{m}$ |
|:------:|:-----------:|:------------:|
| $\vec{R}$ | $+57.1\ \text{m}$ | $-10.5\ \text{m}$ |

This already tells us something important:  The player ends up mostly to the right, and slightly downward from the starting point.


### Step 3: Find the Magnitude of the Resultant

Using the Pythagorean Theorem to get the magnitude:

$$
\begin{aligned}
R^2 &= R_x^2 + R_y^2 \\
&= (57.1\ \text{m})^2 + (10.5\ \text{m})^2 \\
R^2 &= 3416.5\ \text{m}^2 \\
R &= \pm \sqrt{3416.5\ \text{m}^2 } \\
R &= + 58.5\ \text{m} 
\end{aligned}
$$

### Step 4: Find the Direction of the Resultant

To find the direction angle, we can use tangent:

$$
\tan(\theta) = \frac{o}{a} = \frac{10.5}{57.1} \implies \theta = \tan^{-1}\left(\frac{10.5}{57.1}\right) = 10.4^\circ
$$

This angle is measure off of East turned toward the South; draw a picture to convince yourself this is correct. That means the direction is:

$$
10.4^\circ\ \text{South of East}
$$


### Final Answer

The resultant displacement is:

- **Magnitude:** $58.5\ \text{m}$
- **Direction:** $10.4^\circ$ South of East
{% endcapture %}
{% include example.html content=ex %}






## Velocity

In Lecture 01, we spent a lot of time talking about **position** and **displacement**. Now we are ready to introduce the idea of **velocity**, which tells us *how quickly* position changes.

At its core, velocity answers a very simple question:

> How fast, and in what direction, is something changing its position?

This question already hints at something important: velocity is not just about speed. Direction matters.





### What Velocity Means Physically

Velocity describes **how displacement changes with time**. Mathematically, we write this as

$$
\vec{v} = \frac{\Delta \vec{r}}{\Delta t}
$$

where:
- $\Delta \vec{r}$ is the displacement vector, and  
- $\Delta t$ is the time interval over which that displacement occurs.

This definition tells us several important things immediately:

- Velocity depends on **displacement**, not distance.
- Velocity has both a **magnitude** and a **direction**.
- Velocity is therefore a **vector quantity**.

If an object moves quickly but keeps changing direction, its velocity is changing even if its speed stays the same.





### Velocity vs. Speed

In everyday language, the words *speed* and *velocity* are often used interchangeably. In physics, however, they mean different things.

- **Speed** tells you *how fast* something is moving.
- **Velocity** tells you *how fast* something is moving **and in what direction**.

Speed is a scalar. Velocity is a vector.

For example:
- A car traveling at $20\ \text{m/s}$ has a speed of $20\ \text{m/s}$.
- A car traveling at $20\ \text{m/s}$ **north** has a velocity of $20\ \text{m/s north}$.

Throughout this course, we will almost always work with **velocity**, not speed.





### Average Velocity

When we talk about velocity over a finite time interval, we are usually referring to **average velocity**.

Average velocity is defined as

$$
\vec{v}_{\text{avg}} = \frac{\Delta \vec{r}}{\Delta t}
$$

This looks identical to the general definition of velocity, and that is not an accident. Average velocity is simply the displacement divided by the time taken.

A key point to emphasize is that **average velocity depends only on the starting and ending positions**, not on the path taken between them.

This should sound familiar, because it mirrors how **displacement** works.





### Average Velocity Depends on Displacement, Not Distance

Suppose you walk:
- $50\ \text{m}$ east,
- then $50\ \text{m}$ west,

and the entire trip takes $100\ \text{s}$.

Your **total distance traveled** is $100\ \text{m}$, but your **displacement** is zero, because you ended up where you started.

Using the definition of average velocity:

$$
\vec{v}_{\text{avg}} = \frac{0}{100\ \text{s}} = 0
$$

Even though you were clearly moving during the trip, your **average velocity is zero**.

This example highlights an important idea that will come up repeatedly:

> An object can travel a large distance and still have zero average velocity.





### Uniform Velocity

A special and very useful case occurs when an object moves with **uniform velocity**.

Uniform velocity means:
- The object covers equal displacements in equal time intervals.
- Both the magnitude **and** direction of the velocity remain constant.

In this case, the velocity does not change over time.

If an object moves with uniform velocity, then

$$
\vec{v}_{\text{uniform}} = \frac{\Delta \vec{r}}{\Delta t}
$$

for **any** time interval you choose.

Uniform velocity motion will show up repeatedly in our examples because it is mathematically simple and physically intuitive.





### Instantaneous Velocity (Preview)

So far, we have talked about velocity over a **time interval**. But what if we want to know an object’s velocity at a **specific moment**?

For example:
- What is your velocity at exactly noon?
- What is the velocity of a car at the instant it passes a speed limit sign?

This idea leads us to **instantaneous velocity**, which we will explore more carefully in the next sections.

For now, the key idea is this:

> Instantaneous velocity describes motion at a single moment in time, not over a long interval.

We will return to this idea once we begin working with graphs.



