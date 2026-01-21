---
layout: default
title: Pysics for Life Sciences I - Lecture 02
course_home: /courses/phys-for-life-sci-i/
nav_section: lectures
nav_order: 2
---

# Lecture 02 — Velocity




### Trigonometry Review (Right Triangles)

Trigonometry, sometimes casually called “triangle math,” will be used for two main purposes in this course:

- Breaking a diagonal vector’s magnitude (hypotenuse) and direction into a set of horizontal and vertical components.
- Combining horizontal and vertical components of a vector to obtain a single magnitude (hypotenuse) and direction.

Consider a right triangle with:

- One right angle  
- One other angle labeled $ \theta $ ("theta"), which we use as a general symbol for an angle 

shown here:

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Triangle.png' | relative_url }}"
  alt="The image shows a right triangle with the right angle at the lower right corner. The bottom side of the triangle is labeled “Adjacent,” and the vertical side on the right is labeled “Opposite.” The slanted side connecting the bottom left corner to the top right corner is labeled “Hypotenuse.” At the bottom left corner, the angle between the hypotenuse and the adjacent side is marked with the Greek letter theta."
  style="display:block; margin:1.5rem auto; max-width:600px; width:60%;">

Relative to the angle $ \theta $:
- The **hypotenuse** is the side opposite the right angle  
- The **adjacent** side is next to $ \theta $, but not the hypotenuse  
- The **opposite** side is across from $ \theta $  

It is important to note that the "adjacent" side is not always the horizontal side, and the "opposite" side is not always the vertical side. These labels depend entirely on the angle you are referencing.
 






### Breaking a Triangle into Components

In this case, we want to find the opposite and adjacent sides using the hypotenuse and the given angle. This process is often described as “finding the components of the triangle.” To do this, we use cosine and sine:

$$
\cos(\theta) = \frac{\text{Adjacent}}{\text{Hypotenuse}}
\qquad\implies\qquad
\text{Adjacent} = (\text{Hypotenuse}) \times \cos(\theta)
$$

$$
\sin(\theta) = \frac{\text{Opposite}}{\text{Hypotenuse}}
\qquad\implies\qquad
\text{Opposite} = (\text{Hypotenuse}) \times \sin(\theta)
$$

{% capture ex %}
Suppose we have the triangle above with the following measurements: $\theta = 37^\circ$ and a hypotenuse of 65.

We can find the adjacent side using cosine:

$$
\text{Adjacent} = (\text{Hypotenuse}) \times \cos(\theta)
= (65)\cos(37^\circ)
= 52
$$

Similarly, the opposite side is found using sine:

$$
\text{Opposite} = (\text{Hypotenuse}) \times \sin(\theta)
= (65)\sin(37^\circ)
= 39
$$
{% endcapture %}
{% include example.html content=ex %}





 

## Combining Components into Magnitude and Direction

We can also work in reverse. If we know the adjacent and opposite sides, we can find the magnitude and direction of the vector.

### Magnitude (Hypotenuse)

The magnitude comes from the Pythagorean Theorem:

$$
(\text{Hypotenuse})^2 = (\text{Adjacent})^2 + (\text{Opposite})^2
$$

After taking the square root, you will always get a positive and a negative answer. It is your job to decide which makes physical sense. For lengths, the positive value is the meaningful one.

 

### Direction

The direction comes from tangent:

$$
\tan(\theta) = \frac{\text{Opposite}}{\text{Adjacent}}
\qquad\Rightarrow\qquad
\theta = \tan^{-1}\!\left(\frac{\text{Opposite}}{\text{Adjacent}}\right)
$$

Simply giving an angle is not enough to communicate a direction. For example, telling someone to “look 37 degrees” is not helpful unless they also know where to start looking and which way to turn the given 37 degrees. Without a **reference point** and **direction to turn**, the odds of getting the direction right are pretty low.

 

### Direction Requires a Turn-of-Reference

As we just saw, saying “look $ 37^\circ $” alone does not tell someone which way to look. A direction must include:

- A reference direction
- A direction to turn from that reference  


<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/TurnofReference.png' | relative_url }}"
  alt="The image shows a set of horizontal and vertical axes labeled North at the top, South at the bottom, East on the right, and West on the left. A blue arrow points into the upper-right quadrant, and curved arrows around it show two possible angle descriptions labeled “East of North’’ and “North of East.’’ A red arrow points into the lower-left quadrant with curved arrows showing the angle descriptions “South of West’’ and “West of South.’’ A boxed label near the upper left reads “Turn-of-Reference.’’ The diagram illustrates how vector directions can be named relative to different cardinal directions."
  style="display:block; margin:1.5rem auto; max-width:600px; width:60%;">

We will write directions in the form:

$$
\theta = \text{Angle; Turn-of-Reference}
$$

For example is the direction was given as $ 37^\circ $ North of East, the you would:
- Face East
- Turn $ 37^\circ $ toward North  

This direction would be described as **$ 37^\circ $ North of East**.

 
{% capture ex %}
Suppose we have the right triangle from previously with the following measurements: Opposite = 3 and Adjacent = 4. Let’s find the hypotenuse (magnitude) and the direction.

We can find the magnitude using the Pythagorean Theorem:

$$
\begin{aligned}
    (\text{Hypotenuse})^2 &= (\text{Adjacent})^2 + (\text{Opposite})^2 \\
    (\text{Hypotenuse})^2 &= (4)^2 + (3)^2 \\
    (\text{Hypotenuse})^2 &= 16 + 9 \\
    (\text{Hypotenuse})^2 &= 25 \\
    \text{Hypotenuse} &= \pm\sqrt{25} \\
    \text{Hypotenuse} &= \pm 5
\end{aligned}
$$

Since this is the length of a side of a triangle, we take $+5$ as the answer because a negative length does not make sense.

The direction can be found using tangent:

$$
\theta = \tan^{-1}\!\Big( \frac{3}{4} \Big) = 37^\circ
$$

But simply giving this angle is not enough. If we take right to be East and up the page/sceen to be North, then this angle would be referenced using East and then we would turn North. This makes the direction: 

$$
\theta = 37^\circ \text{; North of East}
$$

{% endcapture %}
{% include example.html content=ex %}





{% capture ex %}
## Example: Harder Vector Addition in 2D - Ultimate Frisbee

In this problem, an ultimate Frisbee player runs a pattern made up of three displacement vectors, $\vec{A}$, $\vec{B}$, and $\vec{C}$. We want the **resultant displacement**, meaning the single displacement vector $\vec{R}$ that takes the player from the starting point directly to the ending point.

The given information is:

- $\vec{A} = 10\ \text{m}$ (straight up)
- $\vec{B} = 30\ \text{m}$ (to the right)
- $\vec{C} = 34\ \text{m}$ at an angle of $37^\circ$ below the positive $x$ direction  
  (so it points down and to the right)

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/UltimateFrisbee.png' | relative_url }}"
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