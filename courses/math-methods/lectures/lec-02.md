---
layout: default
title: Mathematical Methods - Lecture 02
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 2
---

# Lecture 02 -- Determinant of a Matrix

One of the unique operations we can perform on a matrix, something we can’t do with just any mathematical object, is calculating the **determinant**. However, the determinant is only defined for **square matrices**, meaning matrices with the same number of rows and columns (an $n \times n$ matrix).

{% include warning.html content="
To repeat this: The derterminant can only be taken for a square ($n \times n$) matrix. 

If you see a $2 \times 3$ or $3 \times 2$ matrix, **no determinant exists**!.
" %}

When a square matrix is viewed as a coordinate transformation (like a rotation, rescaling, or reflection), its determinant tells us something about how that transformation affects “volume” in space. We’re using “volume” loosely here as it means different things depending on the number of dimensions you are avtively working in: area in 2D, volume in 3D, and so on.

For example, a $2 \times 2$ matrix can only act on 2D vectors, so its determinant tells us how **area** changes under the transformation. A $3 \times 3$ matrix, on the other hand, affects **volume** in 3D space. If you’re looking for a vocabulary word, matrices that act this way are called **transformation matrices**. We’ll look at different types of transformations in Lecture 03.

A transformation matrix can do a few different things, the most relivant things right now being an overall rescaling and/or a reflection of objects being transformed. So what does the determinant actually tell us? 

- The **size** of the determinant tells us whether the transformation involves rescaling. For example,
  - if the determinant is 2, then every region of space doubles in volume. 
  - if the determinant is $\pm 1$, then the size of a volume is left unchanged. 
    - In classical mechanics, these are called *volume-preserving* or *incompressible* transformations, like the flow of an ideal fluid.
- The **sign** of the determinant, positive or negative, tells us whether the transformation includes a reflection (also called a *parity flip*, which swaps handedness like right to left or vice versa).
  - if the determinant is *positive*, then no parity flip has occurred.   
  - if the determinant is *negative*, then a parity flip has occurred.   

{% include warning.html content="
A **negative** determinant signals a reflection is part of the overall transformation. That doesn’t mean *only* a reflection occurred, rotations may have happened too, but it does mean something definitely flipped.
" %}


So far, we’ve been talking about what the determinant means. But how do we actually calculate a determinant?

For $2 \times 2$ and $3 \times 3$ matrices, the calculations are pretty straightforward. For larger matrices, we’ll use a method called **cofactor expansion**.

Let’s start with the easy cases first: how to calculate the determinant of $2 \times 2$ and $3 \times 3$ matrices. Then we’ll build up to the more general method from there.






## Determinant of a $2 \times 2$ Matrix

To calculate the determinant of a $2 \times 2$ matrix, start by writing out the matrix. Then, draw a diagonal from the top-left entry to the bottom-right. Remember, this diagonal is called the **main diagonal**. Multiply those two entries to get $a \cdot s$, see the matrix below.

<img
  src="{{ '/courses/math-methods/images/lec02/2by2Det.png' | relative_url }}"
  alt="Diagram illustrating the determinant of a 2×2 matrix. A square bracket encloses four entries labeled a (top left), b (top right), r (bottom left), and s (bottom right). Two diagonals cross the matrix: a blue diagonal from a to s indicating the product a times s, and a red diagonal from b to r indicating the product b times r. The diagram visually emphasizes that the determinant is computed as a times s minus b times r."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

Next, draw a diagonal from the top-right to the bottom-left and multiply those values to get $b \cdot r$.

Now subtract the second diagonal from the first like so $a \cdot s - b \cdot r$, and you are left with the determinant of the $2 \times 2$ matrix:

$$
\det(2\times 2 \text{ matrix}) = \begin{vmatrix}
	a & b \\
	r & s
\end{vmatrix} = as - br
$$

{% include example.html content="
As an example, let's take the determinant of

$$
\mathbf{A} = \begin{bmatrix}
	1 & 2 \\
	-3 & 6
\end{bmatrix}
$$

Using the process we just described, the determinant of $\mathbf{A}$ will be:

$$
\text{det}(\mathbf{A}) = |\mathbf{A}| = \begin{vmatrix}
	1 & 2 \\
	-3 & 6
\end{vmatrix}
= (1)(6) - (2)(-3) = 6 - (-6) = 12
$$

So what does that mean? Since the determinant is *positive*, there's no reflection, and since the magnitude it's *not equal to 1*, it stretches areas (in this 2D case) by a factor of 12. We cannot say anything about rotations, unfortunately.
" %}











## Determinant of a $3 \times 3$ Matrix

The determinant of a $3 \times 3$ matrix is calculated slightly differently than for a $2 \times 2$. As you might have guessed, the method we're using here isn’t general and we'll cover the broader approach shortly. For now, let's focus on learning a trick to calculating the determinant of a $3 \times 3$ matrix.

To compute the determinant of a $3 \times 3$ matrix, start by writing out the matrix:

$$
\begin{bmatrix}
	a & b & c  \\
	r & s & t  \\
	x & y & z  
\end{bmatrix}
$$

Next, repeat the first two columns to the right of the matrix, like this:

<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det1.png' | relative_url }}"
  alt="A diagram showing a 3 by 3 matrix with entries labeled a, b, c in the first row, r, s, t in the second row, and x, y, z in the third row. To the right of the matrix, the first two columns (a, b; r, s; x, y) are repeated. This illustrates the setup step for the diagonal method used to compute a 3 by 3 determinant."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

Start from the top-left entry and draw a diagonal line down and to the right, along the main diagonal again. Then draw two more diagonals that run parallel to it, starting from the second and third entries in the top row. You should get something like this:

<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det2.png' | relative_url }}"
  alt="A 3 by 3 matrix with its first two columns duplicated to the right. Blue arrows indicate three downward diagonals representing the positive determinant terms. The diagonals correspond to the products a times s times z, b times t times x, and c times r times y, which are added together as a part of the determinant calculation."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

Multiply the entries along each of those diagonals and then add them up. This gives:

$$
asz + btx + cry
$$

Now repeat this process for the diagonals that go **up and to the right**, starting from the bottom-left entry and moving up through the matrix:

<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det3.png' | relative_url }}"
  alt="A 3× by 3 matrix with its first two columns duplicated to the right. Red arrows indicate three upward diagonals representing the negative determinant terms. The diagonals correspond to the products x times s times c, y times t times a, and z times r times b, which are added together as part of the determinant calculation."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

Again, multiply the entries along each diagonal and add the results:

$$
xsc + yta + zrb
$$

To get the determinant, simply subtract the second result from the first:

$$
(asz + btx + cry) - (xsc + yta + zrb)
$$

That expression gives the determinant of the general $3 \times 3$ matrix above.

{% capture ex %}
As an example, let's take the determinant of

$$
\mathbf{A} =
\begin{bmatrix}
1 & 2 & 3 \\
-3 & 6 & 4 \\
-1 & -5 & 3
\end{bmatrix}
$$

Using the process we just described, the determinant of $\mathbf{A}$ is:

$$
\text{det}(\mathbf{A}) = |\mathbf{A}| =
\begin{vmatrix}
1 & 2 & 3 \\
-3 & 6 & 4 \\
-1 & -5 & 3
\end{vmatrix}
$$

Now, repeat the first two columns:

<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det4.png' | relative_url }}"
  alt="A diagram showing a numerical 3 by 3 matrix with rows (1, 2, 3), (-3, 6, 4), and (-1, -5, 3). To the right of the matrix, the first two columns (1, 2), (-3, 6), and (-1, -5) are duplicated. This illustrates the setup step for applying the diagonal method to compute the determinant of a 3 by 3 matrix."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

The downward diagonals give:

$$
(1)(6)(3) + (2)(4)(-1) + (3)(-3)(-5) = 18 - 8 + 45 = 55
$$

The upward diagonals give:

$$
(-1)(6)(3) + (-5)(4)(1) + (3)(-3)(2) = -18 - 20 - 18 = -56
$$

Now subtract:

$$
55 - (-56) = 111
$$

So the determinant of $\mathbf{A}$ is 111. That means the matrix does not cause a reflection (since the result is positive), but it does scale volumes up by a factor of 111.
{% endcapture %}
{% include example.html content=ex %}







The determinant of a $3 \times 3$ matrix is calculated differently than that of a $2 \times 2$ matrix. As you may have guessed, the methods we are covering here are not general; we will explore a more general approach after this section.

To compute the determinant of a $3 \times 3$ matrix, start by writing out the matrix.

$$
\begin{bmatrix}
	a & b & c  \\
	r & s & t  \\
	x & y & z  
\end{bmatrix}
$$

Then repeat the first two columns in the last column. This will look something like this:


<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det1.png' | relative_url }}"
  alt="A diagram showing a 3 by 3 matrix with entries labeled a, b, c in the first row, r, s, t in the second row, and x, y, z in the third row. To the right of the matrix, the first two columns (a, b; r, s; x, y) are repeated. This illustrates the setup step for the diagonal method used to compute a 3 by 3 determinant."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

Beginning with the top left element, draw a line down and to the right along the main diagonal. Next, draw another diagonal parallel to this line, starting from the entry in the first row and second column. Finally, draw a third diagonal parallel to the other two, beginning with the element in the first row and third column. This will result in something like this:

<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det2.png' | relative_url }}"
  alt="A 3 by 3 matrix with its first two columns duplicated to the right. Blue arrows indicate three downward diagonals representing the positive determinant terms. The diagonals correspond to the products a times s times z, b times t times x, and c times r times y, which are added together as a part of the determinant calculation."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

Multiply the elements along each of these diagonals individually, and then add the results together. This process will yield $$  asz + btx + cry $$

Next, repeat this process, but start with the bottom left entry and draw a diagonal up and to the right, similar to what you did with the main diagonal. Create two more diagonals, beginning with the next element to the right each time. This will look something like:

<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det3.png' | relative_url }}"
  alt="A 3× by 3 matrix with its first two columns duplicated to the right. Red arrows indicate three upward diagonals representing the negative determinant terms. The diagonals correspond to the products x times s times c, y times t times a, and z times r times b, which are added together as part of the determinant calculation."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

Multiply the elements along each of these diagonals individually, and then add the results for each diagonal. This process will yield $ xsc + yta + zrb $.

Finally, take the first result and subtract the second from it to obtain:

$$
(asz + btx + cry) - (xsc + yta + zrb)
$$

which represents the determinant of the general $3 \times 3$ matrix above.

Please note that what matters here is the **process**, not the specific equation we just derived. Memorizing formulas shouldn't be your primary goal when learning. Instead, focus on understanding the **concepts and reasoning** behind the physical laws and methods; that’s where the real value lies. So don’t spend too much time trying to commit the formula above to memory. Instead, practice the method and get comfortable with the steps involved in the process.


{% capture ex %}
As an example, let's take the determinant of 

$$
\mathbf{A} =
\begin{bmatrix}
1 & 2 & 3 \\
-3 & 6 & 4 \\
-1 & -5 & 3
\end{bmatrix}
$$

Let's begin by duplicating the first tow columns on the right side of the matrix:

<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det4.png' | relative_url }}"
  alt="A diagram showing a numerical 3 by 3 matrix with rows (1, 2, 3), (-3, 6, 4), and (-1, -5, 3). To the right of the matrix, the first two columns (1, 2), (-3, 6), and (-1, -5) are duplicated. This illustrates the setup step for applying the diagonal method to compute the determinant of a 3 by 3 matrix."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

The first set of diagonals, down and to the right, will give: 

$$
(1)(6)(3) + (2)(4)(-1) + (3)(-3)(-5) = 18 - 8 + 45 = 55
$$

and the second set of diagonals, up and to the right, will give: 

$$
(-1)(6)(3) + (-5)(4)(1) + (3)(-3)(2) = -18 - 20 - 18 = -56
$$

Finally, take the first number, from the down and to the right diagonals, and subtract the second number, from the up and to the right diagonals, to get:

$$
55 - (-56) = 111
$$

The determinant of matrix $\mathbf{A}$ is 111. 

What does this mean? The matrix does not parity flip (right handed corrdinates stay right handed), but does scale volumes up by a factor of 111.
{% endcapture %}
{% include example.html content=ex %}








### General Approach: Cofactor Method

Now, let’s explore a general approach to calculating determinants that works consistently, regardless of the size of the square matrix. This method, known as the **cofactor method**, is a powerful technique that applies to *any* square matrix. The basic idea is to break a large matrix into smaller pieces, compute determinants of those smaller matrices, and then combine those results to obtain the determinant of the original matrix.

Let’s start by looking at the general form of a $3 \times 3$ matrix:

$$
\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}
$$

The determinant of $\mathbf{A}$ can be computed by **expanding along a row or a column**. Here, we will expand along the first row (though any row or column may be chosen). When expanding along the first row, the determinant is written as:

$$
\text{det}(\mathbf{A}) = a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}
$$

Here, $a_{1j}$ refers to the element in the first row and $j$-th column, and $C_{1j}$ denotes the corresponding **cofactor**, which we define next.

To compute a cofactor, we first identify the **minor** of a matrix element. The minor of an element $a_{ij}$ is the determinant of the matrix that remains after removing the $i$-th row and $j$-th column from $\mathbf{A}$.

For example, the minor of $a_{11}$ is found by eliminating the first row and first column:

<img
  src="{{ '/courses/math-methods/images/lec02/Cofactor1.png' | relative_url }}"
  alt="A three-by-three matrix labeled with entries a11, a12, and a13 in the top row; a21, a22, and a23 in the middle row; and a31, a32, and a33 in the bottom row. The entire top row is marked, and the middle column is marked, indicating the row and column chosen for a cofactor expansion."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

and taking the determinant of the resulting $2 \times 2$ matrix:

$$
M_{11} = \begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix} = a_{22} a_{33} - a_{23} a_{32}
$$

Similarly, the minor of $a_{12}$ is obtained by eliminating the first row and second column:

<img
  src="{{ '/courses/math-methods/images/lec02/Cofactor2.png' | relative_url }}"
  alt="A three-by-three matrix in which one row and one column are indicated as being removed. The remaining entries represent the smaller matrix, called the minor, that is used when computing a cofactor for a selected matrix element."
  style="display:block; margin:1.5rem auto; max-width:400px; width:50%;">

$$
M_{12} = \begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix} = a_{21} a_{33} - a_{23} a_{31}
$$

The **cofactor** is defined as the signed version of the minor:

$$
C_{ij} = (-1)^{i+j} M_{ij}
$$

For example, the cofactor $C_{11}$ is

$$
C_{11} = (-1)^{1+1} M_{11} = (-1)^2 M_{11} = M_{11}
$$

while the cofactor corresponding to $a_{12}$ is

$$
C_{12} = (-1)^{1+2} M_{12} = (-1)^3 M_{12} = -M_{12}
$$

Following this pattern, we compute each cofactor $C_{1j}$ in the same way.

Once the cofactors are known, we substitute them back into the expansion formula:

$$
\text{det}(\mathbf{A}) = a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}
$$

This procedure generalizes directly to larger matrices. By choosing a row or column and systematically computing cofactors, the determinant of a square matrix of *any* size can be calculated using the same underlying method.

{% capture ex %}
Let’s look at an example to see this process in action. As a helpful note, it is generally easiest to expand along a row or column that contains the most zeros, since this minimizes the number of cofactors you actually need to compute.

Consider calculating the determinant of the matrix:

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 6 \end{bmatrix}
$$

For this first pass, we’ll expand along the first row. For each element in that row, we compute its cofactor by removing the corresponding row and column and finding the determinant of the resulting $2 \times 2$ matrix.

> For the element $1$ (row 1, column 1):
> 
> $$
> \begin{aligned}
> \text{Cofactor of } 1 \, (a_{11}) &= (-1)^{1+1} \det \begin{bmatrix} 4 & 5 \\ 0 & 6 \end{bmatrix} \\
> &= (+1)\big((4)(6) - (5)(0)\big) \\
> &= 24
> \end{aligned}
> $$

> For the element $2$ (row 1, column 2):
> 
> $$
> \begin{aligned}
> \text{Cofactor of } 2 \, (a_{12}) &= (-1)^{1+2} \det \begin{bmatrix} 0 & 5 \\ 1 & 6 \end{bmatrix}\\
> &= -\big((0)(6) - (5)(1)\big) \\
> &= 5
> \end{aligned}
> $$

> For the element $3$ (row 1, column 3):
> 
> $$
> \begin{aligned}
> \text{Cofactor of } 3 \, (a_{13}) &= (-1)^{1+3} \det \begin{bmatrix} 0 & 4 \\ 1 & 0 \end{bmatrix} \\
> &= (+1)\big((0)(0) - (4)(1)\big) \\
> &= -4
> \end{aligned}
> $$

The determinant of $\mathbf{A}$ is found by multiplying each entry in the first row by its cofactor and summing the results:

$$
\det(\mathbf{A}) = (1)(24) + (2)(5) + (3)(-4)
$$

Simplifying gives:

$$
\det(\mathbf{A}) = 24 + 10 - 12 = 22 \implies \det(\mathbf{A}) = 22
$$

---
> **Good tip for the cofactor method:** Expand along rows or columns with lots of zeros!

Let’s redo the same problem, but this time expand along the third row, which contains a zero.

For the element $1$ (row 3, column 1):

> $$
> \begin{aligned}
> \text{Cofactor of } 1 \, (a_{31}) &= (-1)^{3+1} \det \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} \\
> &= (2)(5) - (3)(4) \\
> &= -2
> \end{aligned}
> $$

For the element $0$ (row 3, column 2):

> $$
> \text{Cofactor of } 0 \, (a_{32}) = \text{Doesn't matter — it will be multiplied by 0 anyway.}
> $$

For the element $6$ (row 3, column 3):

> $$
> \begin{aligned}
> \text{Cofactor of } 6 \, (a_{33}) &= (-1)^{3+3} \det \begin{bmatrix} 1 & 2 \\ 0 & 4 \end{bmatrix} \\
> &= (1)(4) - (2)(0) \\
> &= 4
> \end{aligned}
> $$

Putting this together:

$$
\det(\mathbf{A}) = (1)(-2) + (0)C_{32} + (6)(4) = -2 + 0 + 24 = 22
$$

As expected, we obtain the same determinant even though we expanded along a completely different row.
{% endcapture %}
{% include example.html content=ex %}




## Determinant Methods: When to Use Which?
When should you use each of these three determinant methods? The following table helps simply the decision making process:

| **Matrix Size**          | **Best Determinant Method** |
|--------------------------|-----------------------------|
| $2 \times 2$             | &nbsp; Use the diagonal formula: $ad - bc$ |
| $3 \times 3$             | &nbsp; Use diagonal trick **or** cofactor method |
| $4 \times 4$ or larger   | &nbsp; Cofactor method |

Readers looking for a reliable rule of thumb: know the $2 \times 2$ diagonal shortcut ($ad - bc$) and the cofactor method. Those two techniques are enough to calculate the determinant of any square matrix, no matter the size.









## Determinants and Matrix Invertibility

You might be wondering why the determinant matters beyond what it tells us about geometric transformations. One major reason is this: the determinant tells us whether a matrix can be **inverted**. In other words, **whether the matrix has an inverse**.

What do we mean by inverted? Consider the following equation:

$$ 2 x = 4 $$

To solve this we can divide both sides by 2. However, we do not know how to divide by a matrix, we only know how to multiply. Well, dividing both sides by 2 is the same as multipleing both sides by $\frac{1}{2}$, the *inverse* of 2. This gives:

$$ \Big(\frac{1}{2}\Big) \, 2 x = \Big(\frac{1}{2}\Big) \, 4 \qquad\implies\qquad x = 2 $$

and we have solved for $x$. 

To see how this parallels with matrices, consider this system of linear equations:

$$
\begin{aligned}
	2x + 3y &= 7 \\
	-3x + 9y &= -2
\end{aligned} 
\quad \Rightarrow \quad
\begin{bmatrix}
	2 & 3 \\ -3 & 9
\end{bmatrix}
\begin{bmatrix}
	x \\ y
\end{bmatrix} = 
\begin{bmatrix}
	7 \\ -2
\end{bmatrix}
$$

Let's simplify, and generalize, this discussion by writting the current matrix equation as:

$$
\mathbf{A} \vec{r} = \vec{b},
$$

where $\mathbf{A}$ is the coefficient matrix, $\vec{r}$ is the column vector of unknowns, and $\vec{b}$ is the result vector. 


Wouldn’t it be nice if we could just multiply both sides by something the representes the "inverse of $\mathbf{A}$ and instantly solve for $x$ and $y$?. For this to work, we know what ever it is we need to multiply by will need to be a $2 \times 2$ matrix (can you explain why using matrix multiplication rules?), since it will need to act on $\mathbf{A}$.

Let's call $\mathbf{A}^{-1}$ the inverse of $\mathbf{A}$, *if it exists*, since this is paralleling the idea that we are looking for $\frac{1}{\mathbf{A}} = \mathbf{A}^{-1}$. Using $\mathbf{A}^{-1}$ will let us solve for $\vec{r}$ directly via multiplication, similar to how multiplying by $\frac{1}{2}$ solved for $x$ in the previous example. But we must be careful: matrix multiplication is not commutative, so **order matters**. The inverse must be placed so that it acts on $\mathbf{A}$. In this case, that means multiplying on the left:

$$
\begin{aligned}
	\mathbf{A} \vec{r} &= \vec{b} \\
	\mathbf{A}^{-1} \mathbf{A} \vec{r} &= \mathbf{A}^{-1} \vec{b} \\
	\vec{r} &= \mathbf{A}^{-1} \vec{b}
\end{aligned}
$$

where we have used the fact that we are assuming $\mathbf{A}^{-1}$ represents the inverse of $\mathbf{A}$ to allow the simplification: $\mathbf{A}^{-1}\mathbf{A} = 1$. Tehcnically, this operation would yeild the identity matirx $\mathbf{I}$, but we will get to that shortly. The main point for now is to notice how this solution stragety is identical in concept to the simple algebra problem solved previously. 

So how do we know whether a matrix has an inverse and how do we go about finding it? 

The answer to the first part of this quation comes straight from the determinant: **If the determinant is zero, the matrix does not have an inverse.** Why? Because you cannot divide by 0. There are more technical reasons as to why an inverse would not exist for a matrix, which we will get to, but this conceptual justification is good enough for us right now.

Let's check the coefficient matrix we just had. Does this matrix have an inverse? To answer this we calcualte its determinant:

$$
|\mathbf{A}| = \begin{vmatrix}
	2 & 3 \\ -3 & 9
\end{vmatrix}
= (2)(9) - (3)(-3)
= 18 + 9
= 27
$$

Because the determinant is nonzero, this matrix *can* be inverted.

{% include warning.html content="
Not all square matrices have inverses! 

If the determinant is zero, the matrix is said to be *singular* and *can’t be inverted*. 

That means the inverse of this kind of matrix does not exist.
" %}

But, how do we actually find the inverse? One general approach is to use the row-reduction techniques we’ve already learned. Another option is to use a calculator, which is what we’ll rely on for most inverses going forward.







### Row Reduction Method of Finding an Inverse

Let’s walk through the row reduction method once so we can see how it works. After that, we’ll stick to using calculators for finding matrix inverses.

Start by setting up an augmented matrix like this:

$$
\begin{bmatrix}
	2 & 3 & \vline & 1 & 0 \\
	-3 & 9 & \vline & 0 & 1 \\
\end{bmatrix}
$$

The matrix

$$
\begin{bmatrix}
	1 & 0 \\
	0 & 1
\end{bmatrix}
$$

is called the **identity matrix**, $\mathbf{I}$. It works like the number $1$ in multiplication. That means:

$$
\mathbf{A} \mathbf{I} = \mathbf{A} = \mathbf{I} \mathbf{A} 
$$

The side $\mathbf{I}$ acts on does not matter!

To find $\mathbf{A}^{-1}$, we use row reduction to transform the left side of the augmented matrix into the identity matrix. When we succeed, the right side becomes the inverse of $\mathbf{A}$. Here's what the steps look like:

$$
\begin{aligned}
	\begin{bmatrix}
		2 & 3 & \vline & 1 & 0 \\
		-3 & 9 & \vline & 0 & 1 \\
	\end{bmatrix}
	&\implies
	\begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		1 & -3 & \vline & 0 & -\tfrac{1}{3} \\
	\end{bmatrix}
	\\[1ex]
	&\implies
	\begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		0 & -\tfrac{9}{2} & \vline & -\tfrac{1}{2} & -\tfrac{1}{3} \\
	\end{bmatrix}
	\\[1ex]
	&\implies
	\begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		0 & 1 & \vline & \tfrac{1}{9} & \tfrac{2}{27} \\
	\end{bmatrix}
	\\[1ex]
	&\implies
	\begin{bmatrix}
		1 & 0 & \vline & \tfrac{1}{3} & -\tfrac{1}{9} \\
		0 & 1 & \vline & \tfrac{1}{9} & \tfrac{2}{27} \\
	\end{bmatrix}
\end{aligned}
$$

So the inverse of $\mathbf{A}$ is:

$$
\mathbf{A}^{-1} = \frac{1}{27} \begin{bmatrix} 9 & -3 \\ 3 & 2 \end{bmatrix}
$$

{% include result.html content="
This process leads us to a general form for the inverse of a $2 \\times 2$ matrix, 

$$ 
\\mathbf{B} = \\begin{bmatrix}
	b_{11} & b_{12} \\\\ 
	b_{21} & b_{22}
\\end{bmatrix} 
\\implies 
\\mathbf{B}^{-1} = \\frac{1}{\\text{det}(\\mathbf{B})} \\begin{bmatrix}
	b_{22} & -b_{12} \\\\ 
	-b_{21} & b_{11}
\\end{bmatrix} 
$$
" %}

(You should check that this gives the same inverse to the above example on your own!)

How do we confirm that this really is the inverse of $\mathbf{A}$? Multiply them and check if you get the identity matrix:

$$
\begin{aligned}
\mathbf{A}^{-1} \mathbf{A} &= \frac{1}{27}
\begin{bmatrix} 9 & -3 \\ 3 & 2 \end{bmatrix}
\begin{bmatrix} 2 & 3 \\ -3 & 9 \end{bmatrix} \\
&= \frac{1}{27}
\begin{bmatrix}
(9)(2) + (-3)(-3) & (9)(3) + (-3)(9) \\
(3)(2) + (2)(-3) & (3)(3) + (2)(9)
\end{bmatrix} \\
&= \frac{1}{27}
\begin{bmatrix}
27 & 0 \\
0 & 27
\end{bmatrix}
= \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} \\
\Rightarrow \quad \mathbf{A}^{-1} \mathbf{A} &= \mathbf{I}
\end{aligned}
$$

You can also check that $\mathbf{A} \mathbf{A}^{-1}$ gives the identity matrix—it should!


Let’s look at the row reduction method once to understand the process, and then we’ll use a calculator for all inverse matrices from here on.

First, set up your matrix in the following way:

$$ \begin{bmatrix}
	2 & 3 & \vline & 1 & 0 \\
	-3 & 9 & \vline & 0 & 1 \\
\end{bmatrix} $$

The matrix

$$
\begin{bmatrix}
	1 & 0 \\
	0 & 1 
\end{bmatrix}
$$

is called the **identity matrix, $\mathbf{I}$. It functions similarly to the number $1$ in multiplications, acting as the **multiplicative identity** in matrix algebra. This means that when any matrix is multiplied by the identity matrix of the correct dimensions, the original matrix remains unchanged, just like when you multiple a nuber by 1:

$$
\mathbf{A}  \mathbf{I} = \mathbf{I}  \mathbf{A} = \mathbf{A}
$$

To find the inverse of $\mathbf{A}$, we can use row reduction to transform the left side of the augmented matrix above into the identity matrix. The resulting matrix on the right side will be the **inverse of **$\mathbf{A}$, or $\mathbf{A}^{-1}$. Here is how this process looks, step-by-step:

$$
\begin{aligned}
	\begin{bmatrix}
		2 & 3 & \vline & 1 & 0 \\
		-3 & 9 & \vline & 0 & 1 \\
	\end{bmatrix} &\implies \begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		1 & -3 & \vline & 0 & -\frac{1}{3} \\
	\end{bmatrix} \
	&\implies \begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		0 & -\frac{9}{2} & \vline & -\tfrac{1}{2}& -\frac{1}{3} \\
	\end{bmatrix} \\
	& \implies \begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		0 & 1 & \vline & \tfrac{1}{9}& \frac{2}{27} \\
	\end{bmatrix} \\
	\begin{bmatrix}
		2 & 3 & \vline & 1 & 0 \\
		-3 & 9 & \vline & 0 & 1 \\
	\end{bmatrix} &\implies \begin{bmatrix}
		1 & 0 & \vline & \tfrac{1}{3} & -\frac{1}{9} \\
		0 & 1 & \vline & \tfrac{1}{9}& \frac{2}{27} \\
	\end{bmatrix}
\end{aligned}
$$

We claim that the matrix we found is the inverse of $\mathbf{A}$. First, let’s simplify it by factoring out $ \frac{1}{\text{det}(\mathbf{A})} $, giving us:


$$ \mathbf{A}^{-1} = \frac{1}{27} \begin{bmatrix} 9 & -3 \\ 3 & 2 \end{bmatrix} $$


{% include result.html content="
This leads us to a general form for the inverse of a $2 \times 2$ matrix, 
$$ 
\mathbf{B} = \begin{bmatrix}
	b_{11} & b_{12} \\ 
	b_{21} & b_{22}
\end{bmatrix} 
\implies 
\mathbf{B}^{-1} = \frac{1}{\text{det}(\mathbf{B})} \begin{bmatrix}
	b_{22} & -b_{12} \\ 
	-b_{21} & b_{11}
\end{bmatrix} 
$$

You should check this on your own!
" %}


How can we confirm that the matrix we found is indeed the inverse of $\mathbf{A}$? One way is to multiply them together to see if we obtain the identity matrix. Recall that a number multiplied by its inverse should yield 1. We have:

$$
\begin{aligned}
\mathbf{A}^{-1} \mathbf{A} &= \frac{1}{27} \begin{bmatrix} 9 & -3 \\ 3 & 2 \end{bmatrix} \begin{bmatrix}
    2 & 3  \\
    -3 & 9 
\end{bmatrix}  \\
&= \frac{1}{27} \begin{bmatrix}
    (9)(2) + (-3)(-3) & (9)(3) + (-3)(9)\\
    (3)(2) + (2)(-3)  & (3)(3) + (2)(9) 
\end{bmatrix}   \\
&= \frac{1}{27} \begin{bmatrix}
    27 & 0\\
    0  & 27 
\end{bmatrix}  \\
&= \begin{bmatrix}
    1 & 0\\
    0  & 1
\end{bmatrix} \\
\mathbf{A}^{-1} \mathbf{A} &= \mathbf{I}
\end{aligned}
$$

You can verify on your own that the multiplication $\mathbf{A} \mathbf{A}^{-1}$ results in the identity matrix. 








## Other Matrix Operations

Let's look at some other commonly used matrix operations.

### Transpose 

Imagine you're storing data in a spreadsheet. Sometimes you want to swap rows and columns—say, to make a chart. The transpose does exactly that! 

The transpose of a matrix $\mathbf{A}$ is denoted by $\mathbf{A}^T$ and is obtained by flipping the matrix over its main diagonal. This operation switches rows for columns and columns for rows.  Fun Fact: If a matrix equals its transpose, it’s called *symmetric*. Symmetric matrices show up all over physics, especially when your mathematical framework involves the extensive use of tensors, like in general relativity and other field theories. In the element notation, this is a flip of the indices, such that the element at position $a_{ij}$ in matrix $\mathbf{A}$ becomes $a_{ji}$ in matrix $\mathbf{A}^T$. For example, if we have a matrix 

$$
\mathbf{A} = \begin{bmatrix}
	1 & 2 & 3 \\
	4 & 5 & 6
\end{bmatrix}
$$

its transpose is 

$$
\mathbf{A}^T = \begin{bmatrix}
	1 & 4 \\
	2 & 5 \\
	3 & 6
\end{bmatrix}
$$

In physics, the transpose of a matrix is particularly useful in various contexts, such as in linear transformations and systems of equations. For instance, when dealing with vectors and matrices in three-dimensional space, the transpose plays a critical role in changing the representation of a vector from row to column format. This is essential in expressing physical quantities like momentum and force in vector form, where one might need to multiply matrices that represent different physical systems. Moreover, the transpose is utilized in formulating orthogonal matrices, which have applications in rotational transformations and preserving vector lengths during these transformations. We will talk more above this later when we talk about orthogonal matrices in Lecture 05.

### Trace

The trace can only be taken of a square matrix, so let's suppose $\mathbf{A}$ in now a square matrix. The trace of $\mathbf{A}$ is denoted as $\text{Tr}(\mathbf{A})$, is defined as the **sum of the elements on its main diagonal**. The term “trace” comes from the Latin *trahere*, meaning “to draw.” In early matrix notation, it was literally the main diagonal that you’d trace out! 

For a matrix $\mathbf{A}$ of size $n \times n$, the trace is given by:

$$
\text{Tr}(\mathbf{A}) = \sum_{i=1}^{n} a_{ii}
$$

where $a_{ii}$ are the diagonal elements of $\mathbf{A}$. For example, if we have a matrix 

$$
\mathbf{A} = \begin{bmatrix}
	1 & 2 & 3 \\
	4 & 5 & 6 \\
	7 & 8 & 9
\end{bmatrix}
$$

the trace of $\mathbf{A}$ is 

$$
\text{Tr}(\mathbf{A}) = 1 + 5 + 9 = 15
$$

The trace has important applications in physics and engineering, particularly in the study of linear operators and quantum mechanics. The trace of an operator in matrix form is related to the sum of its eigenvalues, which can provide insights into the properties of the system being analyzed. For example, in quantum mechanics, the trace is used to compute expected values and to describe the behavior of quantum states under transformations. Additionally, in thermodynamics, the trace of the density matrix is used to find the partition function, which is fundamental in statistical mechanics for calculating the thermodynamic properties of systems. 


## Application: Inertia Matrix

Consider a rigid body with a distribution of mass defined by the following mass elements at specified coordinates in 3D space. The inertia tensor $\mathbf{I}$ (something you will see in Classical Mechanics) is given by:

$$
\mathbf{I} = \begin{bmatrix}
	4 & 0 & 0 \\
	0 & 3 & 1 \\
	0 & 1 & 2
\end{bmatrix}
$$

a) Calculate the Determinant of $\mathbf{I}$.
	
> Using the co-factor method along the first row gives: 
>		
>		$$ \begin{vmatrix}
>			4 & 0 & 0 \\
>			0 & 3 & 1 \\
>			0 & 1 & 2
>		\end{vmatrix} = 4 \, \begin{vmatrix}
>		3 & 1 \\ 1 & 2
>		\end{vmatrix} + 0 \, C_{12} + 0 \, C_{13} = 4 ( 6 - 1 ) = 20 $$
	
b) Calculate the Trace of $\mathbf{I}$.
	
>		Summing the main diagonal gives:
>		
>		$$ \text{Tr}\left(\begin{bmatrix}
>			4 & 0 & 0 \\
>			0 & 3 & 1 \\
>			0 & 1 & 2
>		\end{bmatrix}\right) = 4 + 3 + 2 = 9 $$
	
c) Calculate the Transpose of $\mathbf{I}$.
	
>		The transpose would be:
>		
>		$$ \begin{bmatrix}
>			4 & 0 & 0 \\
>			0 & 3 & 1 \\
>			0 & 1 & 2
>		\end{bmatrix}^{T} = \begin{bmatrix}
>		4 & 0 & 0 \\
>		0 & 3 & 1 \\
>		0 & 1 & 2
>		\end{bmatrix} $$
>		
>		**When the transpose is the same as the original the matrix is called symmetric!**
	
d) Calculate the Inverse of $\mathbf{I}$.
	
>		And the inverse can be found using the row reduction method:
>		    
>           $$
>			\begin{aligned}
>				\begin{bmatrix}
>					4 & 0 & 0 & \vline & 1 & 0 & 0 \\
>					0 & 3 & 1 & \vline & 0 & 1 & 0\\
>					0 & 1 & 2 & \vline & 0 & 0 & 1
>				\end{bmatrix} &\implies \begin{bmatrix}
>					1 & 0 & 0 &\vline & \tfrac{1}{4}  & 0 & 0 \\
>					0 & 1 & 2 &\vline & 0 & 0 & 1\\
>					0 & 3 & 1 &\vline & 0 & 1 & 0
>				\end{bmatrix} \\
>				&\implies \begin{bmatrix}
>					1 & 0 & 0 &\vline & \tfrac{1}{4}  & 0 & 0 \\
>					0 & 1 & 2 &\vline & 0 & 0 & 1\\
>					0 & 0 & -5 &\vline & 0 & 1 & -3
>				\end{bmatrix}  \\
>				&\implies \begin{bmatrix}
>					1 & 0 & 0 &\vline & \tfrac{1}{4}  & 0 & 0 \\
>					0 & 1 & 2 &\vline & 0 & 0 & 1\\
>					0 & 0 & 1 &\vline & 0 & -\tfrac{1}{5} & \tfrac{3}{5}
>				\end{bmatrix}   \\
>				\begin{bmatrix}
>					4 & 0 & 0 & \vline & 1 & 0 & 0 \\
>					0 & 3 & 1 & \vline & 0 & 1 & 0\\
>					0 & 1 & 2 & \vline & 0 & 0 & 1
>				\end{bmatrix}  &\implies \begin{bmatrix}
>					1 & 0 & 0 &\vline & \tfrac{1}{4}  & 0 & 0 \\
>					0 & 1 & 0 &\vline & 0 & \tfrac{2}{5} & - \tfrac{1}{5}\\
>					0 & 0 & 1 &\vline & 0 & -\tfrac{1}{5} & \tfrac{3}{5}
>				\end{bmatrix}
>			\end{aligned}

	So, the inverse of $\mathbf{I}$ is:
	 
>	$$ 
>    \mathbf{I}^{-1} = \begin{bmatrix}
>		 \tfrac{1}{4}  & 0 & 0 \\
>		 0 & \tfrac{2}{5} & - \tfrac{1}{5}\\
>		 0 & -\tfrac{1}{5} & \tfrac{3}{5}
>	\end{bmatrix} = \frac{1}{20} \begin{bmatrix}
>	5  & 0 & 0 \\
>	0 & 8 & - 4 \\
>	0 & -4 & 12
>	\end{bmatrix} $$





## Problem:


- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.


Consider a three-dimensional stress tensor $\boldsymbol{\sigma}$, which describes the state of stress at a point in a material under loading conditions. The stress tensor is defined as follows:

$$
\boldsymbol{\sigma} = \begin{bmatrix}
	4 & 2 & 0 \\
	2 & 5 & 1 \\
	0 & 1 & 3
\end{bmatrix}
$$

The stress tensor (a fancy way of saying matrix in most situations) tells us about how the materials is compressed/stretched or is being twisted. 

a) Calculate the Determinant of $\mathbf{\sigma}$. (The determinant of the stress tensor provides insight into the state of stress in the material and can be used to assess stability.). 
b) alculate the Trace of $\mathbf{\sigma}$. (The trace of the stress tensor represents the sum of the normal stresses acting on the material, which can influence material behavior under load.)  
c) Calculate the Transpose of $\mathbf{\sigma}$. (The transpose of the stress tensor helps verify its properties, particularly since the stress tensor must be symmetric for physical applications. Is the given stress tensor symmetric? How do you know?)  
d) Calculate the Inverse of $\mathbf{\sigma}$. (The inverse of the stress tensor can be useful for transformations in continuum mechanics, especially in determining the relationship between stress and strain.)  





