---
layout: default
title: Physics for Life Sciences I - Lecture 01
course_home: /courses/phys-for-life-sci-i/
nav_section: lectures
nav_order: 1
---

# Lecture 01 — Math Review and Vectors



## Why Trigonometry Shows Up in Physics

In the sciences, especially in physics, we frequently work in situations where direction is a vital part of the description. For example, if you’re on a boat traveling out to a site to perform sonar readings, you need to know where you’re starting and, from there, where you’re headed. That means you need both the distance to the site *and* the direction you must travel to get there. In this example, we’re talking about the *position* of the site relative to the starting point, such as the dock where the boat is stored.


<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Position1.png' | relative_url }}"
  alt="The image shows a simple diagram of a dock on the lower left and a sonar site on the upper right. A straight purple arrow points from the dock toward the sonar site, representing the position of the site relative to the starting point. The dock is labeled “Dock,” and the sonar site is labeled “Sonar Site.” The arrow is labeled “Position,” indicating both the distance and the direction from the dock to the sonar site."
  style="display:block; margin:1.5rem auto; max-width:600px; width:60%;">


At other times, we deal with characteristics that don’t require a direction to be fully described. For instance, *mass* is a measure of the amount of stuff an object is made of. The kilogram is, quite literally, tied to a specific number of silicon atoms, though we’re not concerned with that level of detail here. What does matter is that the amount of stuff in an object doesn’t depend on which way the object is facing. 

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/MassExample.png' | relative_url }}"
  alt="The image shows three cartoon rock-like characters with faces. On the left, one character says, “Roary has a new diet trend!” In the upper right, a second character, Roary, is facing North and has a speech bubble that says, “Facing North I have a mass of 12 silicon atoms!” In the lower right, Roary is shown facing East and has a speech bubble that says, “Now that I have turned East, I have a mass of 10 silicon atoms!” The figures illustrate the idea of an object humorously claiming to change mass by simply rotating."
  style="display:block; margin:1.5rem auto; max-width:800px; width:100%;">

It wouldn’t make sense for something to have a mass of 10 atoms when facing East but somehow be made of 12 atoms when facing North. That just doesn’t make sense.

Anytime direction matters, trigonometry becomes a useful tool. In particular, right triangles let us move back and forth between:

- a single magnitude (hypotenuse) and direction, and  
- separate horizontal and vertical components.  

We will use this idea constantly when working with vectors.





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



## Scalars and Vectors

Physical quantities in science fall into two broad categories: scalars and vectors.

### Scalars

A **scalar** is any quantity that only needs a magnitude to be fully described.

Examples include:
- Mass  
- Energy  
- Temperature  

Scalars do not have not need a direction to be fully defined.

 

### Vectors

A **vector** is any quantity that requires both a magnitude and a direction to be fully described.

Examples include:
- Displacement  
- Velocity  
- Force  

We will be working with vector quantities for this lecture and throughout most of the course.

 

#### Visual Representation of Vectors

Vectors are drawn as arrows:

- The **length** of the arrow represents the magnitude  
- The **direction** of the arrow represents the direction  

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Vector.png' | relative_url }}"
  alt="A vector represented by an arrow lebeled A pointing up and to the right."
  style="display:block; margin:1.5rem auto; max-width:600px; width:60%;">

Only these two features matter; the position the arrow is drawn on the page does not.

When writing vectors symbolically, we place a little arrow over the variable:

$$
\vec{A}
$$

to both indicate the name of the vector and to remind us that it is, in fact, a vector. For example, the above symbol and the previous figure would be read as “the vector $ A $.”

Vectors can come in all direction and lengths. For instance, here are a could of different vectors: 

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Vectors.png' | relative_url }}"
  alt="The image shows three separate vectors. On the left, a long arrow labeled “A” points up and to the right. In the center, a shorter arrow labeled “B” also points up and to the right but at a shallower angle. On the right, a long arrow labeled “C” points down and to the right. Each vector label appears just to the left of its corresponding arrow."
  style="display:block; margin:1.5rem auto; max-width:600px; width:60%;">

Observing the three vectors we can see that $\vec{A}$ and $\vec{B}$ point in the same direction and $\vec{C}$ points in the opposite direction. Similarly, we can see that $\vec{A}$ and $\vec{C}$ have the same length and $\vec{B}$ is half the length of the other two vectors. 

Mathematically, if one vector is twice as long as another and points in the same direction we would write this as:

$$
\vec{B} = \tfrac{1}{2} \vec{A}
$$

However, if two vectors have the same length but point in opposite directions, we use a minus sign to flip the direction without changing the length:

$$
\vec{C}  = -\vec{A}
$$

As a result, vectors can differ by:
- Magnitude, or  
- Direction, or  
- Both  

 



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


### Displacement Is NOT the Same as Distance

Distance and displacement are often confused, but they are not the same thing. Distance measures how much ground you cover as you move along a path. It depends on the route taken and adds up all motion, including turns, detours, and backtracking. This is what the odometer of a car measures and what a step counter on your smart watch would read.

Displacement, on the other hand, *ignores the path entirely*. It is the straight-line vector pointing from where you started to where you ended. The actualy path does not matter!

In physics, we almost always care about displacement rather than distance. Unless you are explicitly told otherwise, you should assume that displacement is the quantity being used.


### A Walking Example to Introduce Vector Addition

Generally we will work with multiple displacements that get added together to find a total displacement for a trip, or something. 

To help exaplain this, let’s go on a quick journey together. This will be a mental one, unless you feel the urge to actually try it yourself, in which case please follow the script. 

Suppose you were just at your instructor’s office asking questions about the difference between distance and displacement. Your instructor says, “Follow me, and remember to wear your step counter.” Your step counter is set to measure how far your have walked along in your journey, in meters. In other words, the step counter measures the  *distance* you have traveled.

First, you walk with your instructor from their office to a room in the Arts and Sciences building. The step counter keeps ticking upward the entire time, recording every step along the path you take. From there, you walk to the campus coffee shop, again following the hallways, staircases, and doors needed to get there. By the time you arrive,  *your step counter displays the \textbf{total distance*} you walked from your instructor's office to the coffee shop. That number is your  *total distance traveled*. Distance depends on the exact path you took, including all turns, stairs, and detours along the way.

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/DisplacmentDistance.png' | relative_url }}"
  alt="The diagram shows three locations connected by arrows. On the left is a point labeled “Instructor’s Office (start).” From this point, a solid arrow goes up and to the right to a point labeled “Arts & Sciences room,” representing one displacement. From the Arts & Sciences room, a second solid arrow goes down and to the right to a point labeled “Coffee Shop (end),” representing another displacement. A long solid arrow runs directly from the Instructor’s Office to the Coffee Shop, showing the total displacement for the trip. Dashed blue lines show the actualy winding path you took from one location to the next. The figure illustrates how discplancement is difference from distance and how individual displacements add together to produce a single resultant displacement from start to end."
  style="display:block; margin:1.5rem auto; max-width:800px; width:100%;">

Now consider your  *displacement*. Displacement does not care about the path you followed. Instead, the displacement for the first leg of your trip is the straight-line vector that points from your starting location, the instructor’s office, directly to your final location, the Arts and Sciences building. From there, the second leg of your journey, the displacement points in a straight-line from the Arts and Sciences building towards the coffee shop. 

The *total displacement* of your adventure points from your starting location, the office, to the final location, the coffee shop, and can be found by adding the first leg of your trip and the second to ge the total. This vector has both a magnitude, how far apart those two locations are, and a direction, which way the coffee shop is relative to the office. Even though you may have walked through hallways, around corners, or up and down stairs, your total displacement only depends on where you started and where you ended.

In the figure above, the total length of the dashed path represents the route you actually walked. the length of this path woulf give you the  *distance* you traveled, or how many steps you took in total. The solid arrows, drawn from one point to the next, represent the  *displacements* for each part of the trip.

These two quantities are not the same. Distance depends on the path you take, while displacement depends only on where you start and where you end. For this reason, the dashed path and the solid arrows generally have different lengths.

In physics, distance is used far less often than displacement. In fact, unless you are specifically told otherwise, you should work with displacement rather than distance.







### Adding Displacements

The previous example helps illustrate how displacements add together (more on this in a second). The displacement from the office to the Arts and Sciences room is one vector. The displacement from that room to the coffee shop is another vector. If you add those two displacement vectors together, the result is a single displacement vector that points directly from the office to the coffee shop. That total displacement is exactly the same as if you had drawn one arrow from the start of the trip to the end.

This is why displacement is a vector quantity and why vector addition matters. Distance tells you how much ground you covered, while displacement tells you where you ended up relative to where you started.








## Vector Addition (Tail-to-Tip Method)

To add vectors graphically, we use the **tail-to-tip** method:

1. Place the tail of the second vector at the tip of the first vector.
2. Draw a new vector from the tail of the first vector to the tip of the last vector.
3. This new vector is the *resultant*.

Symbolically, if vectors $\vec{A}$ and $\vec{B}$ are added to produce vector $\vec{R}$, the resultant, we write:

$$
\vec{A} + \vec{B} = \vec{R}
$$

and this looks like:

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Tailtotip.png' | relative_url }}"
  alt="The diagram illustrates the tail-to-tip method of vector addition. Vector A is drawn from a starting point up and to the right. Vector B begins at the tip of vector A and points down and to the right. A dashed arrow labeled R connects the tail of vector A to the tip of vector B, representing the resultant vector. A blue circle highlights the point where the tail of vector B meets the tip of vector A, emphasizing the tail-to-tip placement used when adding vectors."
  style="display:block; margin:1.5rem auto; max-width:800px; width:60%;">

The order of addition does not matter:

$$
\vec{A} + \vec{B} = \vec{R} \quad\text{and}\quad \vec{B} + \vec{A} = \vec{R}
$$

as we can see in this figure:

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Ordernotmatter.png' | relative_url }}"
  alt="The diagram shows two vectors, labeled A and B, arranged in two different orders. In the top arrangement, vector A is drawn first and vector B is placed tail-to-tip with it, forming a bent path. A dashed arrow labeled R connects the starting point of the first vector to the ending point of the second vector, representing the resultant displacement. In the lower arrangement, the order of the vectors is reversed, with vector B drawn first and vector A placed tail-to-tip afterward. The dashed resultant arrow R is the same in both cases, showing that the final displacement does not depend on the order in which the vectors are added."
  style="display:block; margin:1.5rem auto; max-width:800px; width:60%;">

The same idea works for more than two vectors. If several displacements occur one after another, as pictured here:

<img
  src="{{ '/courses/phys-for-life-sci-i/images/lec01/Multiplevectoraddition.png' | relative_url }}"
  alt="The diagram shows several vectors labeled A, B, C, and D arranged head-to-tail in a zigzag path. Vector A starts on the left and points up and to the right. Vector B begins at the tip of vector A and points down and to the right. Vector C starts at the tip of vector B and points slightly up and to the right, and vector D continues upward from the tip of vector C. A dashed arrow labeled R runs directly from the starting point of vector A to the ending point of vector D, representing the resultant vector. The figure illustrates how multiple displacement vectors add together to produce a single overall resultant displacement."
  style="display:block; margin:1.5rem auto; max-width:800px; width:60%;">

Their sum gives the total displacement:

$$
\vec{A} + \vec{B} + \vec{C} + \vec{D} = \vec{R}
$$

where, again, $\vec{R}$ is the resultant displacement and points from the starting point to the final point.




## Vector Addition (Algebraic Method)

The pictorial method of adding vectors is very helpful for building intuition, but it quickly becomes impractical when we want to do actual calculations. For that, we use the **algebraic method** of vector addition.

The key idea behind the algebraic method is simple: vectors can only be added component by component. More specifically, the most important rules are:

- Horizontal **can** add to Horizontal.
- Vertical **can** add to Vertical.
- Horizontal **cannot** add to Vertical.

This means we need to keep track of, or calculate the, horizontal and vertical components of vectors separately throughout our problem.

An effective way to stay organized is to place all vector components into a table. This makes it easy to see which quantities can be added together, which quantites you still need to figure out, and helps prevent simple mistakes.

Below is a general-purpose table structure that we will use repeatedly.

| Vector | Horizontal Component (x) | Vertical Component (y) |
|:------:|:--------------------------:|:------------------------:|
| $\vec{A}$ | $A_x$ | $A_y$ |
| $\vec{B}$ | $B_x$ | $B_y$ |
| $\vec{C}$ | $C_x$ | $C_y$ |
|------|--------------------------|------------------------|
| $\vec{R}$ | $R_x$ | $R_y$ |

where $\vec{R} = \vec{A} + \vec{B} + \vec{C}$, and you just add or remove more rows as needed.

In this table:
- Each row represents a single vector.
- The horizontal column contains all $x$-components.
    - You will need to pick which horizontal direction is positive. For consistently, we will always assume right/East is positive.
- The vertical column contains all $y$-components.
    - You will need to pick which vertical direction is positive. For consistently, we will always assume up-the-page/North is positive.
- The final row represents the summed components of the resultant vector $\vec{R}$.

Because vectors can only be added add component by component, we write:

$$
R_x = A_x + B_x + C_x + \dots
$$

and

$$
R_y = A_y + B_y + C_y + \dots
$$

Once we have $R_x$ and $R_y$, we can then:
1. Find the magnitude of the resultant vector using the Pythagorean Theorem.
2. Find the direction using inverse tangent and turn-of-reference language.

since these components represent the horizontal and vertical sides of a right triangle. 



{% capture ex %}
Suppose a person walks:
- $5$ meters to the right
- then $3$ meters to the left

We can represent these two displacements as vectors:

- $\vec{A} = 5\ \text{m to the right}$
- $\vec{B} = 3\ \text{m to the left}$

However, isn't "left" just "negative right"? In this case we can rewrite the second vector to get:

- $\vec{A} = +5\ \text{m; right}$
- $\vec{B} = -3\ \text{m; right}$

where $-3$ meters to the right means go 3 meters to the left. 

Now, let's build our table:

| Vector | Right | Up |
|:------:|:--------------------------:|:------------------------:|
| $\vec{A}$ | $+5$ | $0$ |
| $\vec{B}$ | $-3$ | $0$ |
|------|--------------------------|------------------------|
| $\vec{R}$ | $+2$ | $0$ |

and we have the resultant displacement:

- $2\ \text{m}$ to the right

Even though the person walked a *total distance* of $8\ \text{m}$, their **displacement** depends only on the start and end positions.

### Why This Is Useful

This example highlights several important ideas:

- Vectors include direction, which we represent using signs in one dimension.
- Opposite directions subtract automatically when vectors are added.
- Displacement is different from distance, even in simple cases.
{% endcapture %}
{% include example.html content=ex %}



{% capture ex %}
Now let’s look at a slightly more interesting case where the two vectors point in **different directions**.

Suppose a person walks:
- $4$ meters to the East
- then $3$ meters to the North

We can represent these two displacements as vectors:

- $\vec{A} = 4\ \text{m to the East}$
- $\vec{B} = 3\ \text{m to the North}$

First let's choose out positive horizontal and vertical directions:
- Horizontal: East
- Vertical: North

This is the first, and last, time we will explicity list our positive directions. From now on we will assume Right/East and Up/North are the positive directions. Putting this into a table, we have:

| Vector | East | North |
|:------:|:-----------:|:------------:|
| $\vec{A}$ |  |  |
| $\vec{B}$ |  |  |
|:------:|:-----------:|:------------:|
| $\vec{R}$ |  |  |

Using these positive directions, we can fill in this table to get:


| Vector | East (x) | North (y) |
|:------:|:-----------:|:------------:|
| $\vec{A}$ | $+4\ \text{m}$ | $0$ |
| $\vec{B}$ | $0$ | $+3\ \text{m}$ |
|:------:|:-----------:|:------------:|
| $\vec{R}$ | $+4\ \text{m}$ | $+3\ \text{m}$ |

You shoud vertify that you agree with these rtesults.

So the components of the resultant displacement $\vec{R}$ are:
- $R_x = +4\ \text{m}$
- $R_y = +3\ \text{m}$

### Finding the Magnitude of the Resultant

The magnitude of the resultant comes from combining the horizontal and vertical components of the resultant vector. This can *only* be done using Pythagorean Theorem:

$$
R = \sqrt{R_x^2 + R_y^2}
$$

to get:

$$
R = \sqrt{(4)^2 + (3)^2} = \sqrt{16 + 9} = 5\ \text{m}
$$


### Finding the Direction of the Resultant

To find the direction, we use tangent:

$$
\tan(\theta) = \frac{\text{North}}{\text{East}} = \frac{3}{4}
$$

and taking the inverse:

$$
\theta = \tan^{-1}\!\left(\frac{3}{4}\right) = 37^\circ
$$

If you sketch out the resultant vector, and its components, you will see it is a right triangle with the angle we found being given off the horizontal. This means the reference direction is East and we have to turn North, giving the full direction as:

- $37^\circ$ **North of East**

### Final Result

The resultant displacement is:
- **Magnitude:** $5\ \text{m}$
- **Direction:** $37^\circ$ North of East


### Why This Is Useful

This example reinforces several important ideas:

- Vectors in different directions must be broken into components before adding.
- Horizontal and vertical components are handled separately.
- The magnitude and direction of a resultant vector come from its components.
- Direction must be stated using both an angle and a turn-of-reference.
{% endcapture %}
{% include example.html content=ex %}