---
layout: default
title: Pysics for Life Sciences I - Lecture 01
course_home: /courses/phys-for-life-sci-i/
nav_section: lectures
nav_order: 1
---

# Lecture 01 — Math Review and Vectors (Part 1)

## Why Trigonometry Shows Up in Physics

In physics, we often care not just about how much of something there is, but also about which way it points. Anytime direction matters, trigonometry becomes a useful tool. In particular, right triangles let us move back and forth between:

- A single magnitude and direction  
- Separate horizontal and vertical components  

We will use this idea constantly when working with vectors.

 

### Trigonometry Review (Right Triangles)

Consider a right triangle with:
- One right angle  
- One other angle labeled $ \theta $ (theta), which we use as a general symbol for an angle  

Relative to the angle $ \theta $:
- The **hypotenuse** is the side opposite the right angle  
- The **adjacent** side is next to $ \theta $, but not the hypotenuse  
- The **opposite** side is across from $ \theta $  

These labels depend entirely on which angle you choose. The “adjacent” side is not always horizontal, and the “opposite” side is not always vertical.

 

### Breaking a Triangle into Components

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

 

# Combining Components into Magnitude and Direction

We can also work in reverse. If we know the adjacent and opposite sides, we can find the magnitude and direction of the vector.

## Magnitude (Hypotenuse)

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

 





