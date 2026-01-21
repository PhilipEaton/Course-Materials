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

$\vec{A}$ points straight up with magnitude $10\ \text{m}$.

So its components are:

- $A_x = 0\ \text{m}$
- $A_y = +10\ \text{m}$

#### Vector $\vec{B}$

$\vec{B}$ points to the right with magnitude $30\ \text{m}$.

So its components are:

- $B_x = +30\ \text{m}$
- $B_y = 0\ \text{m}$

#### Vector $\vec{C}$

$\vec{C}$ has magnitude $34\ \text{m}$ and is $37^\circ$ below the positive $x$ direction.

That means:
- The horizontal component is positive (to the right)
- The vertical component is negative (down)

So:

$$
C_x = (34\ \text{m})\cos(37^\circ)
$$

$$
C_y = -(34\ \text{m})\sin(37^\circ)
$$

Numerically:

$$
C_x = (34)\cos(37^\circ) = 27.1\ \text{m}
$$

$$
C_y = -(34)\sin(37^\circ) = -20.5\ \text{m}
$$


### Step 2: Add Components to Get the Resultant Components

The resultant vector $\vec{R}$ is:

$$
\vec{R} = \vec{A} + \vec{B} + \vec{C}
$$

Add the $x$-components:

$$
R_x = A_x + B_x + C_x = 0 + 30 + 27.1 = 57.1\ \text{m}
$$

Add the $y$-components:

$$
R_y = A_y + B_y + C_y = 10 + 0 - 20.5 = -10.5\ \text{m}
$$

So the resultant components are:

- $R_x = +57.1\ \text{m}$
- $R_y = -10.5\ \text{m}$

This already tells us something important:  
The player ends up mostly to the right, and slightly downward from the starting point.


### Step 3: Find the Magnitude of the Resultant

Use the Pythagorean Theorem:

$$
R^2 = R_x^2 + R_y^2
$$

$$
R^2 = (57.1\ \text{m})^2 + (10.5\ \text{m})^2
$$

$$
R^2 = 3416.5\ \text{m}^2
$$

$$
R = \sqrt{3416.5\ \text{m}^2} = 58.5\ \text{m}
$$

So the magnitude of the resultant displacement is:

$$
R = 58.5\ \text{m}
$$


### Step 4: Find the Direction of the Resultant

To find the direction angle, use tangent:

$$
\tan(\theta) = \frac{|R_y|}{R_x} = \frac{10.5}{57.1} = 0.184
$$

So:

$$
\theta = \tan^{-1}(0.184) = 10.4^\circ
$$

Now we need to describe the direction using turn-of-reference language.

We have:
- $R_x$ is positive (east/right)
- $R_y$ is negative (south/down)

So the vector points slightly **south of east**.

That means the direction is:

$$
10.4^\circ\ \text{South of East}
$$


### Final Answer

The resultant displacement is:

- **Magnitude:** $58.5\ \text{m}$
- **Direction:** $10.4^\circ$ South of East
{% endcapture %}
{% include example.html content=ex %}