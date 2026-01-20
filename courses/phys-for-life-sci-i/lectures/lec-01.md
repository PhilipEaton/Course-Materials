---
layout: default
title: Pysics for Life Sciences I - Lecture 01
course_home: /courses/phys-for-life-sci-i/
nav_section: lectures
nav_order: 1
---

# Lecture 01 — Math Review and Vectors (Part 1)



## Why Trigonometry Shows Up in Physics

In the sciences, especially in physics, we frequently work in situations where direction is a vital part of the description. For example, if you’re on a boat traveling out to a site to perform sonar readings, you need to know where you’re starting and, from there, where you’re headed. That means you need both the distance to the site \textit{and} the direction you must travel to get there. In this example, we’re talking about the \textit{position} of the site relative to the starting point, such as the dock where the boat is stored.


<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Position1.png' | relative_url }}"
  alt="The image shows a simple diagram of a dock on the lower left and a sonar site on the upper right. A straight purple arrow points from the dock toward the sonar site, representing the position of the site relative to the starting point. The dock is labeled “Dock,” and the sonar site is labeled “Sonar Site.” The arrow is labeled “Position,” indicating both the distance and the direction from the dock to the sonar site."
  style="display:block; margin:1.5rem auto; max-width:600px; width:60%;">


At other times, we deal with characteristics that don’t require a direction to be fully described. For instance, \textit{mass} is a measure of the amount of stuff an object is made of. The kilogram is, quite literally, tied to a specific number of silicon atoms, though we’re not concerned with that level of detail here. What does matter is that the amount of stuff in an object doesn’t depend on which way the object is facing. It wouldn’t make sense for something to have a mass of 10 atoms when facing East but somehow be made of 12 atoms when facing North. That just doesn’t make sense.

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/MassExample.png' | relative_url }}"
  alt="The image shows three cartoon rock-like characters with faces. On the left, one character says, “Roary has a new diet trend!” In the upper right, a second character, Roary, is facing North and has a speech bubble that says, “Facing North I have a mass of 12 silicon atoms!” In the lower right, Roary is shown facing East and has a speech bubble that says, “Now that I have turned East, I have a mass of 10 silicon atoms!” The figures illustrate the idea of an object humorously claiming to change mass by simply rotating."
  style="display:block; margin:1.5rem auto; max-width:800px; width:100%;">



In physics, we often care not just about how much of something there is, but also about which way it points. Anytime direction matters, trigonometry becomes a useful tool. In particular, right triangles let us move back and forth between:

- A single magnitude and direction  
- Separate horizontal and vertical components  

We will use this idea constantly when working with vectors.

 

### Trigonometry Review (Right Triangles)

Trigonometry, sometimes casually called “triangle math,” will be used for two main purposes in this course:

- Breaking a diagonal vector’s magnitude and direction into a set of horizontal and vertical components.
- Combining horizontal and vertical components of a vector to obtain a single magnitude and direction.

Consider the following right triangle:

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Triangle.png' | relative_url }}"
  alt="The image shows a right triangle with the right angle at the lower right corner. The bottom side of the triangle is labeled “Adjacent,” and the vertical side on the right is labeled “Opposite.” The slanted side connecting the bottom left corner to the top right corner is labeled “Hypotenuse.” At the bottom left corner, the angle between the hypotenuse and the adjacent side is marked with the Greek letter theta."
  style="display:block; margin:1.5rem auto; max-width:600px; width:60%;">


The hypotenuse, which is the side opposite the right angle, is called the magnitude when the triangle represents a vector. The angle in the bottom left corner, $\theta$ (pronounced “theta”), helps us identify which sides are the “opposite’’ and the “adjacent.’’ The opposite side is the one across from the given angle, and the adjacent side is the one next to the angle that is not the hypotenuse.

It is important to note that the adjacent side is not always the horizontal side, and the opposite side is not always the vertical side. These labels depend entirely on the angle you are using.

Consider a right triangle with:
- One right angle  
- One other angle labeled $ \theta $ (theta), which we use as a general symbol for an angle  

Relative to the angle $ \theta $:
- The **hypotenuse** is the side opposite the right angle  
- The **adjacent** side is next to $ \theta $, but not the hypotenuse  
- The **opposite** side is across from $ \theta $  

These labels depend entirely on which angle you choose. The “adjacent” side is not always horizontal, and the “opposite” side is not always vertical.

 

### Breaking a Triangle into Components

In this case, we want to find the opposite and adjacent sides using the hypotenuse and the given angle. This process is often described as “finding the components of the triangle.” To do this, we use cosine and sine:

\[
\cos(\theta) = \frac{\text{Adjacent}}{\text{Hypotenuse}}
\qquad\implies\qquad
\text{Adjacent} = (\text{Hypotenuse}) \times \cos(\theta)
\]

\[
\sin(\theta) = \frac{\text{Opposite}}{\text{Hypotenuse}}
\qquad\implies\qquad
\text{Opposite} = (\text{Hypotenuse}) \times \sin(\theta)
\]

{% capture ex %}
Suppose we have the triangle above with the following measurements: $\theta = 37^\circ$ and a hypotenuse of 65.

We can find the adjacent side using cosine:

\[
\text{Adjacent} = (\text{Hypotenuse}) \times \cos(\theta)
= (65)\cos(37^\circ)
= 52
\]

Similarly, the opposite side is found using sine:

\[
\text{Opposite} = (\text{Hypotenuse}) \times \sin(\theta)
= (65)\sin(37^\circ)
= 39
\]
{% endcapture %}
{% include example.html content=ex %}

If the hypotenuse represents a vector, then it corresponds to the **magnitude** of that vector. The adjacent and opposite sides represent the vector’s components.

Using trigonometry:

$$
\cos(\theta) = \frac{\text{Adjacent}}{\text{Hypotenuse}}
\qquad\Rightarrow\qquad
\text{Adjacent} = (\text{Hypotenuse})\cos(\theta)
$$

$$
\sin(\theta) = \frac{\text{Opposite}}{\text{Hypotenuse}}
\qquad\Rightarrow\qquad
\text{Opposite} = (\text{Hypotenuse})\sin(\theta)
$$

This process is often described as **finding the components** of a vector.

 

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

However, an angle by itself is usually **not descriptive enough**.

 

### Direction Requires a Turn-of-Reference

Saying “$ 37^\circ $” alone does not tell someone which way to look. A direction must include:
- A reference direction
- A direction to turn from that reference  

We write directions in the form:

$$
\theta = \text{Angle; Turn-of-Reference}
$$

For example:
- Face East
- Turn $ 37^\circ $ toward North  

This direction would be described as **$ 37^\circ $ North of East**.

This idea will come up repeatedly when working with vectors in two dimensions.

 

## Scalars and Vectors

In physics, quantities fall into two broad categories.

## Scalars

A **scalar** is any quantity that only needs a magnitude to be fully described.

Examples include:
- Mass  
- Energy  
- Temperature  

Scalars do not have direction.

 

### Vectors

A **vector** is any quantity that requires both a magnitude and a direction.

Examples include:
- Displacement  
- Velocity  
- Force  

Vectors are the main focus of this lecture and much of the course.

 

## Visual Representation of Vectors

Vectors are drawn as arrows:
- The **length** of the arrow represents the magnitude  
- The **direction** of the arrow represents the direction  

Only these two features matter. The position of the arrow on the page does not.

 

## Vector Notation and Comparison

When writing vectors symbolically, we place an arrow over the variable:

$$
\vec{A}
$$

This is read as “the vector $ A $.”

If one vector is twice as long as another and points in the same direction:

$$
\vec{B} = 2\vec{A}
$$

A minus sign flips the direction but does not change the magnitude:

$$
-\vec{A}
$$

This vector has the same length as $ \vec{A} $, but points in the opposite direction.

As a result, vectors can differ by:
- Magnitude  
- Direction  
- Or both  

 



## Displacement

The first vector quantity we will work with is called **displacement**.

Displacement is the straight-line distance and direction from an initial position to a final position. It does not depend on the path taken between those two points. Only the starting location and the ending location matter.

Displacement is a vector, which means it has:
- A magnitude (how far apart the two points are)
- A direction (which way the final position is relative to the starting position)

We typically represent displacement using the symbol:

$$
\Delta x
$$

The Greek letter $\Delta$ (delta) means “change in,” and $x$ is commonly used to represent position. Written mathematically:

$$
\Delta x = x_{\text{final}} - x_{\text{initial}}
$$


## Displacement Is NOT the Same as Distance

Distance and displacement are often confused, but they are not the same thing.

Distance measures how much ground you cover as you move along a path. It depends on the route taken and adds up all motion, including turns, detours, and backtracking.

Displacement, on the other hand, ignores the path entirely. It is the straight-line vector pointing from where you started to where you ended.

In physics, we almost always care about displacement rather than distance. Unless you are explicitly told otherwise, you should assume that displacement is the quantity being used.


## A Walking Example

Imagine starting at your instructor’s office. From there, you walk to a room in the Arts and Sciences building, and then continue on to the campus coffee shop.

Your step counter records every step along the way. The total number of steps corresponds to the **distance** you traveled.

Now consider your **displacement**. Your displacement is the single straight-line vector that points directly from the instructor’s office to the coffee shop. It does not care about the hallways, staircases, or turns you made along the way.

Even though your distance traveled is fairly large, your displacement may be much smaller.


## Adding Displacements

Displacements can be added together because they are vectors.

- The displacement from the office to the Arts and Sciences room is one vector.
- The displacement from that room to the coffee shop is a second vector.

If you place these two vectors head-to-tail, the result is a single vector that points from the office directly to the coffee shop. This final vector is called the **resultant displacement**.


## Vector Addition (Tail-to-Tip Method)

To add vectors graphically, we use the **tail-to-tip** method:

1. Place the tail of the second vector at the tip of the first vector.
2. Draw a new vector from the tail of the first vector to the tip of the last vector.
3. This new vector is the resultant.

Symbolically, if vectors $\vec{A}$ and $\vec{B}$ are added to produce vector $\vec{C}$, we write:

$$
\vec{A} + \vec{B} = \vec{C}
$$

The order of addition does not matter:

$$
\vec{A} + \vec{B} = \vec{B} + \vec{A}
$$

This is why vector addition is often described as being **commutative**.


## Multiple Displacements

The same idea works for more than two vectors. If several displacements occur one after another, their sum gives the total displacement:

$$
\vec{A} + \vec{B} + \vec{C} + \vec{D} = \vec{R}
$$

Here, $\vec{R}$ is the resultant displacement from the starting point to the final point.

This is exactly how your brain naturally keeps track of where you end up, even if you take a complicated path to get there.





## Full Vector Addition Problem (Ultimate Frisbee)

In this problem, an ultimate Frisbee player runs a pattern made up of three displacement vectors, $\vec{A}$, $\vec{B}$, and $\vec{C}$. We want the **resultant displacement**, meaning the single displacement vector $\vec{R}$ that takes the player from the starting point directly to the ending point.

The given information is:

- $\vec{A} = 10\ \text{m}$ (straight up)
- $\vec{B} = 30\ \text{m}$ (to the right)
- $\vec{C} = 34\ \text{m}$ at an angle of $37^\circ$ below the positive $x$ direction  
  (so it points down and to the right)

We want:
1. The magnitude of $\vec{R}$
2. The direction of $\vec{R}$ using turn-of-reference language


## Step 1: Break Each Vector into Components

We will treat right as $+x$ and up as $+y$.

### Vector $\vec{A}$

$\vec{A}$ points straight up with magnitude $10\ \text{m}$.

So its components are:

- $A_x = 0\ \text{m}$
- $A_y = +10\ \text{m}$

### Vector $\vec{B}$

$\vec{B}$ points to the right with magnitude $30\ \text{m}$.

So its components are:

- $B_x = +30\ \text{m}$
- $B_y = 0\ \text{m}$

### Vector $\vec{C}$

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


## Step 2: Add Components to Get the Resultant Components

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


## Step 3: Find the Magnitude of the Resultant

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


## Step 4: Find the Direction of the Resultant

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


## Final Answer

The resultant displacement is:

- **Magnitude:** $58.5\ \text{m}$
- **Direction:** $10.4^\circ$ South of East



