---
layout: default
title: Mathematical Methods - Lecture 02
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 2
---

# Lecture 02 -- Determinant of a Matrix

One of the unique operations we can perform involving a matrix, which we cannot do with other objects, is to calculate its **determinant**. The determinant can only be calculated for a **square matrix**, meaning a matrix with an equal number of rows and columns, or an \( n \times n \) matrix.

{% include warning.html content="
The derterminant can only be taken for a square ($n \times n$) matrix. 

If you see a $2 \times 3$ or $3 \times 2$ matrix, **no determinant exists**!}.
" %}

When a square matrix is interpreted as a transformation of coordinates—such as a rotation, rescaling, or reflection—the determinant reveals the effect of that transformation on a "volume" in the corresponding space. Here, ``volume” is in quotes because its meaning changes depending on the $n$-dimensions you are working within: area in 2D, volume in 3D, and so on. 

For instance, if the square matrix is \(2 \times 2\), it can only act on a 2-dimensional vector. Thus, the determinant of a \(2 \times 2\) matrix indicates how the **area** is changed by the transformation represented by the matrix. Similarly, if the matrix is \(3 \times 3\), the determinant describes how a **3-dimensional volume** is affected by the transformation represented by the matrix. For those looking for a new vocabulary word, matrices that represent a coordinate transformation are called **transformation matrices**. We will spend time addressing each of these transformations in Lecture 03. 

Because a matrix can transform space, its determinant reveals whether that transformation involves an overall rescaling and/or a reflection. 

A rescaling of the volume of space occurs depending on the size of the determinant of the transformation matrix. 

- If the determinant of a transformation matrix is \(\pm 1\), then the transformation represented by the matrix preserves volume, meaning the size of any region in space stays the same after the transformation.     
    - In classical mechanics transformations that preserve volume are called *volume-preserving* or *incompressible*, such as the flow of an ideal fluid.
- If the absolute value of the determinant is *not* \(1\), then the transformation scales the volume of space by that factor. 
    - For example, if the determinant is \(2\), then any region of space will be stretched so its volume doubles. 

A reflection of the space is signaled by the determinant of the transformation matrix being negative. This is called a *parity flip*, meaning you changed the handedness from right to left or vice versa.

The determinant does not specify if any rotations occur as the result of a transformation, but it is generally safe to assume rotations are involved in almost all transformations.

{% include warning.html content="
A negative determinant doesn’t always mean a reflection alone has occurred; rotations could still have taken place as well. But if the sign is negative, then you know something definately was flipped. 
" %}

With all this talk of determinants, it would be nice if we actually knew how to calculate them. The calculation of a determinant of a $2 \times 2$ and $3 \times 3$ matrix is fairly straightforward, but for larger matrices we generally use the **method of cofactors**. 

Let's first see how the determinant is calculated for $2 \times 2$ and $3 \times 3$ matrices, and then look at the more general approach. 







## Determinant of a $2 \times 2$ Matrix

To calculate the determinant of a \(2 \times 2\) matrix, start by writing out the matrix. Next, draw a diagonal line from the top left entry to the bottom right entry; this diagonal is known as the **main diagonal**, which we will discuss further below. Multiply the two elements on this diagonal. In the example below, this would yield \(a\cdot s\).

Then, draw another diagonal line from the top right element to the bottom left element and multiply the two numbers to obtain \(b\cdot r\). This process looks something like:


<img
  src="{{ '/courses/math-methods/images/lec02/2by2Det.png' | relative_url }}"
  alt="Diagram illustrating the determinant of a 2×2 matrix. A square bracket encloses four entries labeled a (top left), b (top right), r (bottom left), and s (bottom right). Two diagonals cross the matrix: a blue diagonal from a to s indicating the product a times, and a red diagonal from b to r indicating the product b times r. The diagram visually emphasizes that the determinant is computed as a times s minus b times r."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

Then you take the first diagonal and subtract off the second diagonal: $a\cdot s - b\cdot r = as-br$ and that is the determinant of a $2 \times 2$ matrix:

\[
\det(2\times 2 \text{ matrix}) = \begin{vmatrix}
	a & b \\ r & s
\end{vmatrix} = a s - b r 
\]


{% include example.html content="
As an example, let's take the determinant of 

$$ \mathbf{A} = \begin{bmatrix}
	1 & 2 \\ -3& 6
\end{bmatrix} $$

Using the process described above, the determinant of $\mathbf{A}$ will be:

$$ \text{det}(\mathbf{A}) = |\mathbf{A}| = \begin{vmatrix}
	1 & 2 \\ -3& 6
\end{vmatrix}  = (1)(6) - (2)(-3) = 6 - (-6) = 12 $$

From the value of the determinant we can see that this matrix will not cause any reflections (it is positive), but does scale areas (since it is only 2 dimensional) up by a factor of 12. We cannot say anything about rotations, unfortunately.
" %}













## Determinant of a $3 \times 3$ Matrix

The determinant of a \(3 \times 3\) matrix is calculated differently than that of a \(2 \times 2\) matrix. As you may have guessed, the methods we are covering here are not general; we will explore a more general approach after this section.

To compute the determinant of a \(3 \times 3\) matrix, start by writing out the matrix.

\[
\begin{bmatrix}
	a & b & c  \\
	r & s & t  \\
	x & y & z  
\end{bmatrix}
\]

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

Multiply the elements along each of these diagonals individually, and then add the results for each diagonal. This process will yield \( xsc + yta + zrb \).

Finally, take the first result and subtract the second from it to obtain:
\[
(asz + btx + cry) - (xsc + yta + zrb)
\]
which represents the determinant of the general \(3 \times 3\) matrix above.


{% include example.html content="
As an example, let's take the determinant of 
$$ \mathbf{A} = \begin{bmatrix}
	1 & 2 & 3 \\ -3& 6 & 4 \\ -1 & -5 & 3
\end{bmatrix} $$

Using the process we just described, the determinant of $\mathbf{A}$ will be:

$$ \text{det}(\mathbf{A}) = |\mathbf{A}| = \begin{vmatrix}
	1 & 2 & 3 \\ -3& 6 & 4 \\ -1 & -5 & 3
\end{vmatrix}  $$

Let's repeat the first two columns:

<img
  src="{{ '/courses/math-methods/images/lec02/3by3Det4.png' | relative_url }}"
  alt="A diagram showing a numerical 3 by 3 matrix with rows (1, 2, 3), (-3, 6, 4), and (-1, -5, 3). To the right of the matrix, the first two columns (1, 2), (-3, 6), and (-1, -5) are duplicated. This illustrates the setup step for applying the diagonal method to compute the determinant of a 3 by 3 matrix."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

The first set of diagonals, down and to the right, will give: 

$$ (1)(6)(3) + (2)(4)(-1) + (3)(-3)(-5) = 18 - 8 + 45 = 55$$

and the second set of diagonals, up and to the right, will give: 

$$ (-1)(6)(3) + (-5)(4)(1) + (3)(-3)(2) = -18 - 20 -18 = -56$$

Finally, take the first number, from the down and to the right diagonals, and subtract the second number, from the up and to the right diagonals, to get:

$$ 55 - (-56) = 111 $$

The determinant of matrix $\mathbf{A}$ is 111. The matrix does not appear to cause any reflections, but does scale volumes up by a factor of 111.
" %}












### General Approach: Cofactor Method

Now, let’s explore a general approach to calculating determinants that works consistently, regardless of the number of dimensions of the square matrix. This method, known as the **cofactor method**, is a powerful technique applicable to any square matrix. The cofactor method involves breaking down a larger matrix into smaller parts, calculating the determinants of these smaller matrices, and then combining them to find the determinant of the original matrix.

Let's start by looking at the general form of a $3 \times 3$ matrix:
\[
\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}
\]

The determinant of $\mathbf{A}$ can be calculated by **expanding along a row or a column**. Here, we’ll expand along the first row (though any row or column could be chosen). The formula for the determinant of a $3 \times 3$ matrix expanded along the first row is:

\[
\text{det}(\mathbf{A}) = a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}
\]

where \( a_{1j} \) represents the elements of the matrix in the first row and the $j$-th column, and \( C_{1j} \) represents the **cofactor** of the $a_{ij}$ element, to be defined next.

To calculate each cofactor, we start by identifying the **minor** of each element. The minor of a matrix element \( a_{ij} \) is the determinant of the matrix that remains after removing the \( i \)-th row and \( j \)-th column from \( \mathbf{A} \).

For example, the minor of \( a_{11} \) is can be found by eliminating the first row and first column:


<img
  src="{{ '/courses/math-methods/images/lec02/Cofactor1.png' | relative_url }}"
  alt="A three-by-three matrix labeled with entries a11, a12, and a13 in the top row; a21, a22, and a23 in the middle row; and a31, a32, and a33 in the bottom row. The entire top row is marked, and the middle column is marked, indicating the row and column chosen for a cofactor expansion."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

and we take the determinant of that:

$$
M_{11} = \begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix} = a_{22} a_{33} - a_{23} a_{32} 
$$

and the minor of \( a_{12} \) is found by eliminating the first row and the second column:

<img
  src="{{ '/courses/math-methods/images/lec02/Cofactor2.png' | relative_url }}"
  alt="A three-by-three matrix in which one row and one column are indicated as being removed. The remaining entries represent the smaller matrix, called the minor, that is used when computing a cofactor for a selected matrix element."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">



$$
M_{12} = \begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix} = a_{21} a_{33} - a_{23} a_{31} 
$$


The **cofactor** is defined as the signed minor in the following manner:

$$ C_{ij} = (-1)^{i+j} M_{ij} $$


For example, the cofactor \( C_{11} \) is then given by:

\[
C_{11} = (-1)^{1+1} M_{11} = (-1)^{2} M_{11} =  M_{11}
\]

Similarly, the cofactor for \( a_{12} \) is:

\[
C_{12} = (-1)^{1+2} M_{12} = (-1)^{3} M_{12}= -M_{12}
\]

Following this pattern, we calculate each cofactor \( C_{1j} \) in this manner.

With the cofactors calculated, we substitute back into the formula:

\[
\text{det}(\mathbf{A}) = a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}
\]

This approach generalizes to larger matrices as well, where we can choose any row or column for expansion. By systematically calculating cofactors, the determinant of matrices of any size can be computed in the exact same manner.


{% include example.html content="
Let's look at an example of this process. As a helpfulf note, it is generally easiest to choose to work along the row or column with the most zeros to minimize your work when using the cofactor method. This example will help show how this trick works. Consider taking the determinate of the matrix:

\[
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 6 \end{bmatrix}
\]

For this example, let's expand along the first row. For each element in the first row, we find the cofactor by removing the row and column of that element and calculating the determinant of the resulting \(2 \times 2\) submatrix.

For the element \(1\) (row 1, column 1):

\[
\text{Cofactor of } 1 \, (a_{11}) = (-1)^{1+1} \det \begin{bmatrix} 4 & 5 \\ 0 & 6 \end{bmatrix} = \det \begin{bmatrix} 4 & 5 \\ 0 & 6 \end{bmatrix} = (4)(6) - (5)(0) = 24
\]

For the element \(2\) (row 1, column 2):

\[
\text{Cofactor of } 2 \,(a_{12}) = (-1)^{1+2} \det \begin{bmatrix} 0 & 5 \\ 1 & 6 \end{bmatrix} = -\det \begin{bmatrix} 0 & 5 \\ 1 & 6 \end{bmatrix} = -\big((0)(6) - (5)(1)\big) = 5
\]

For the element \(3\) (row 1, column 3):

\[
\text{Cofactor of } 3 \, (a_{13}) = (-1)^{1+3} \det \begin{bmatrix} 0 & 4 \\ 1 & 0 \end{bmatrix} = \det \begin{bmatrix} 0 & 4 \\ 1 & 0 \end{bmatrix} = (0)(0) - (4)(1) = -4
\]

The determinant of \(\mathbf{A}\) is the sum of each element in the first row multiplied by its respective cofactor:

\[
\det(\mathbf{A}) = (1)(24) + (2)(5) + (3)(-4)
\]

Simplifying this gives the determinate of $\mathbf{A}$:

\[
\det(\mathbf{A}) = 24 + 10 - 12 = 22 \implies  \det(\mathbf{A}) = 22
\]

> **Good tip for the cofactor method: Expand along rows or columns with lots of zeros!**

Let's redo the previous example, but expand along row 3, since it has a zero. This gives:

For the element \(1\) (row 2, column 1):

\[
\text{Cofactor of } 1 \, (a_{31}) = (-1)^{3+1} \det \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} = (2)(5) - (3)(4) = -2
\]

For the element \(0\) (row 3, column 2):

\[
\text{Cofactor of } 0  \, (a_{32}) = \text{Doesn't matter since it will be multiplied by 0 later!}
\]

For the element \(6\) (row 3, column 3):

\[
\text{Cofactor of } 6  \, (a_{33})= (-1)^{3+3} \det \begin{bmatrix} 1 & 2 \\ 0 & 4 \end{bmatrix} = (1)(4) - (2)(0) = 4
\]

The determinant of \(\mathbf{A}\) is the sum of each element in the first row multiplied by its respective cofactor:

\[
\det(\mathbf{A}) = (1)(-2) + (0)C_{32} + (6)(4) = -2 + 0 + 24 = 22 \implies \det(\mathbf{A}) = 22
\]

and we get the same thing even though we expanded along a totally different row!
" %}







## Determinant Methods: When to Use Which?
When should you use each of these three determinant methods? The following table helps simply the decision making process:

| **Matrix Size** | **Best Determinant Method** |
|---------------|-----------------------------|
| $2 \times 2$ | Use the diagonal formula: $ad - bc$ |
| $3 \times 3$ | Use diagonal trick **or** cofactor method |
| $4 \times 4$ or larger | Cofactor method |

Readers looking for specific recommendations, then know the determinant of a $2 \times 2$ matrix diagonal formula $ad - bc$, and the cofactor method. Those two techniques are all you need to calculate the determinant of any sized matrix. 



## Determinants and Matrix Invertibility


You may be wondering why the determinant is important beyond what it reveals about geometric transformations (which we will explore further in the next lecture). One key reason is that the determinant tells us if a matrix can be **inverted**—that is, **if a matrix has an inverse**.

Consider the following system of linear equations:

\[
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
\]

Wouldn't it be helpful if we could multiply both sides by some matrix and immediately solve for \( x \) and \( y \)? If we set this up as a matrix equation, we would be looking at solving something like this:

\[
\mathbf{A} \vec{r} = \vec{b}
\]

where \( \mathbf{A} \) is the matrix of coefficients, \( \vec{r} \) is the column vector of unknowns, and \( \vec{b} \) is the result vector. An inverse matrix, if it exists, would allow us to solve for \( \vec{r} \) directly in the following manner.  Remember, the order of multiplication for matrices matters! So, make sure that the inverse of \( \mathbf{A} \) is positioned correctly to act on \( \mathbf{A} \), whether on the left or the right.:

{\allowdisplaybreaks
\begin{align*}
	\mathbf{A} \vec{r} &= \vec{b} \\[0.75ex]
	\mathbf{A}' \mathbf{A} \vec{r} &= \mathbf{A}' \vec{b} \\[0.75ex]
	\underbrace{\mathbf{A}' \mathbf{A}}_\text{= 1} \vec{r} &= \mathbf{A}' \vec{b} \\[0.75ex]
	\vec{r} &= \underbrace{\mathbf{A}' \vec{b} }_\text{Answer}
\end{align*}}

In this case, \( \mathbf{A}' \) **would represent the inverse of** \( \mathbf{A} \).

\begin{center}
	Notationally, the inverse of $\mathbf{A}$ is written as $\mathbf{A}^{-1}$.
\end{center}

Why do we call it the inverse? Think of it this way: in algebra, if you have \( 2x = 4 \), you would multiply both sides by the inverse of \( 2 \), which is \( \frac{1}{2} \), to solve for \( x \). So, $\frac{1}{2}$ is the inverse of $2$, which makes a lot of sense! 

Now, how can we determine if a matrix can be inverted? It’s simple: check if its determinant is zero. **If the determinant is zero, the matrix does not have an inverse**

{% include warning.html content="
Not all square matrices have inverses! **If the determinant is zero**, the matrix is said to be *singular* and \textbf{can't be inverted
" %}



For example, the following matrix:

$$ |\mathbf{A}| = \begin{vmatrix}
	2 & 3 \\ -3 & 9
\end{vmatrix} = (2)(9) - (3)(-3) = 18 + 9 = 27 $$

can be inverted (or has an inverse) since its determinant is non-zero. But how do we actually find the inverse? One general method is to use the row reduction techniques we learned previously. Another option is to simply use a calculator, which we’ll rely on for finding most inverses in the future.





### Row Reduction Method of Finding an Inverse

Let’s look at the row reduction method once to understand the process, and then we’ll use a calculator for all inverse matrices from here on.

First, set up your matrix in the following way:

$$ \begin{bmatrix}
	2 & 3 & \vline & 1 & 0 \\
	-3 & 9 & \vline & 0 & 1 \\
\end{bmatrix} $$

The matrix

\[
\begin{bmatrix}
	1 & 0 \\
	0 & 1 
\end{bmatrix}
\]

is called the **identity matrix, $\mathbf{I}$. It functions similarly to the number \(1\) in multiplications, acting as the **multiplicative identity** in matrix algebra. This means that when any matrix is multiplied by the identity matrix of the correct dimensions, the original matrix remains unchanged, just like when you multiple a nuber by 1:

\[
\mathbf{A}  \mathbf{I} = \mathbf{I}  \mathbf{A} = \mathbf{A}
\]

To find the inverse of $\mathbf{A}$, we can use row reduction to transform the left side of the augmented matrix above into the identity matrix. The resulting matrix on the right side will be the **inverse of **$\mathbf{A}$, or $\mathbf{A}^{-1}$. Here is how this process looks, step-by-step:

{\renewcommand{\arraystretch}{1.25}
\begin{align*}
	\begin{bmatrix}
		2 & 3 & \vline & 1 & 0 \\
		-3 & 9 & \vline & 0 & 1 \\
	\end{bmatrix} &\implies \begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		1 & -3 & \vline & 0 & -\frac{1}{3} \\
	\end{bmatrix} \\[1.15ex]
	&\implies \begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		0 & -\frac{9}{2} & \vline & -\tfrac{1}{2}& -\frac{1}{3} \\
	\end{bmatrix} \\[1.15ex]
	& \implies \begin{bmatrix}
		1 & \tfrac{3}{2} & \vline & \tfrac{1}{2} & 0 \\
		0 & 1 & \vline & \tfrac{1}{9}& \frac{2}{27} \\
	\end{bmatrix} \\[1.15ex]
	\begin{bmatrix}
		2 & 3 & \vline & 1 & 0 \\
		-3 & 9 & \vline & 0 & 1 \\
	\end{bmatrix} &\implies \begin{bmatrix}
		1 & 0 & \vline & \tfrac{1}{3} & -\frac{1}{9} \\
		0 & 1 & \vline & \tfrac{1}{9}& \frac{2}{27} \\
	\end{bmatrix}
\end{align*}}

We claim that the matrix we found is the inverse of $\mathbf{A}$. First, let’s simplify it by factoring out \( \frac{1}{\text{det}(\mathbf{A})} \), giving us:


$$ \mathbf{A}^{-1} = \frac{1}{27} \begin{bmatrix} 9 & -3 \\ 3 & 2 \end{bmatrix} $$


{% include result.html content="
This leads us to a general form for the inverse of a \(2 \times 2\) matrix, 
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


How can we confirm that the matrix we found is indeed the inverse of \(\mathbf{A}\)? One way is to multiply them together to see if we obtain the identity matrix. Recall that a number multiplied by its inverse should yield 1. We have:

$$
\begin{aligned}
\mathbf{A}^{-1} \mathbf{A} &= \frac{1}{27} \begin{bmatrix} 9 & -3 \\ 3 & 2 \end{bmatrix} \begin{bmatrix}
    2 & 3  \\
    -3 & 9 
\end{bmatrix}  \\[1.15ex]
&= \frac{1}{27} \begin{bmatrix}
    (9)(2) + (-3)(-3) & (9)(3) + (-3)(9)\\
    (3)(2) + (2)(-3)  & (3)(3) + (2)(9) 
\end{bmatrix}   \\[1.15ex]
&= \frac{1}{27} \begin{bmatrix}
    27 & 0\\
    0  & 27 
\end{bmatrix}  \\[1.15ex]
&= \begin{bmatrix}
    1 & 0\\
    0  & 1
\end{bmatrix} \\[1.15ex]
\mathbf{A}^{-1} \mathbf{A} &= \mathbf{I}
\end{aligned}
$$

You can verify on your own that the multiplication \(\mathbf{A} \mathbf{A}^{-1}\) results in the identity matrix. 








## Other Matrix Operations

Let's look at some other commonly used matrix operations.

### Transpose 

Imagine you're storing data in a spreadsheet. Sometimes you want to swap rows and columns—say, to make a chart. The transpose does exactly that! 

The transpose of a matrix \(\mathbf{A}\) is denoted by \(\mathbf{A}^T\) and is obtained by flipping the matrix over its main diagonal. This operation switches rows for columns and columns for rows.  Fun Fact: If a matrix equals its transpose, it’s called *symmetric*. Symmetric matrices show up all over physics, especially when your mathematical framework involves the extensive use of tensors, like in general relativity and other field theories. In the element notation, this is a flip of the indices, such that the element at position \(a_{ij}\) in matrix \(\mathbf{A}\) becomes \(a_{ji}\) in matrix \(\mathbf{A}^T\). For example, if we have a matrix 

\[
\mathbf{A} = \begin{bmatrix}
	1 & 2 & 3 \\
	4 & 5 & 6
\end{bmatrix}
\]

its transpose is 

\[
\mathbf{A}^T = \begin{bmatrix}
	1 & 4 \\
	2 & 5 \\
	3 & 6
\end{bmatrix}
\]

In physics, the transpose of a matrix is particularly useful in various contexts, such as in linear transformations and systems of equations. For instance, when dealing with vectors and matrices in three-dimensional space, the transpose plays a critical role in changing the representation of a vector from row to column format. This is essential in expressing physical quantities like momentum and force in vector form, where one might need to multiply matrices that represent different physical systems. Moreover, the transpose is utilized in formulating orthogonal matrices, which have applications in rotational transformations and preserving vector lengths during these transformations. We will talk more above this later when we talk about orthogonal matrices in Lecture 05.

### Trace

The trace can only be taken of a square matrix, so let's suppose \(\mathbf{A}\) in now a square matrix. The trace of $\mathbf{A}$ is denoted as \(\text{Tr}(\mathbf{A})\), is defined as the **sum of the elements on its main diagonal**. The term “trace” comes from the Latin *trahere*, meaning “to draw.” In early matrix notation, it was literally the main diagonal that you’d trace out! 

For a matrix \(\mathbf{A}\) of size \(n \times n\), the trace is given by:

\[
\text{Tr}(\mathbf{A}) = \sum_{i=1}^{n} a_{ii}
\]

where \(a_{ii}\) are the diagonal elements of \(\mathbf{A}\). For example, if we have a matrix 

\[
\mathbf{A} = \begin{bmatrix}
	1 & 2 & 3 \\
	4 & 5 & 6 \\
	7 & 8 & 9
\end{bmatrix}
\]

the trace of \(\mathbf{A}\) is 

\[
\text{Tr}(\mathbf{A}) = 1 + 5 + 9 = 15
\]

The trace has important applications in physics and engineering, particularly in the study of linear operators and quantum mechanics. The trace of an operator in matrix form is related to the sum of its eigenvalues, which can provide insights into the properties of the system being analyzed. For example, in quantum mechanics, the trace is used to compute expected values and to describe the behavior of quantum states under transformations. Additionally, in thermodynamics, the trace of the density matrix is used to find the partition function, which is fundamental in statistical mechanics for calculating the thermodynamic properties of systems. 


## Application: Inertia Matrix

Consider a rigid body with a distribution of mass defined by the following mass elements at specified coordinates in 3D space. The inertia tensor $\mathbf{I}$ (something you will see in Classical Mechanics) is given by:

\[
\mathbf{I} = \begin{bmatrix}
	4 & 0 & 0 \\
	0 & 3 & 1 \\
	0 & 1 & 2
\end{bmatrix}
\]

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
>				\end{bmatrix} \\[1.15ex]
>				&\implies \begin{bmatrix}
>					1 & 0 & 0 &\vline & \tfrac{1}{4}  & 0 & 0 \\
>					0 & 1 & 2 &\vline & 0 & 0 & 1\\
>					0 & 0 & -5 &\vline & 0 & 1 & -3
>				\end{bmatrix}  \\[1.15ex]
>				&\implies \begin{bmatrix}
>					1 & 0 & 0 &\vline & \tfrac{1}{4}  & 0 & 0 \\
>					0 & 1 & 2 &\vline & 0 & 0 & 1\\
>					0 & 0 & 1 &\vline & 0 & -\tfrac{1}{5} & \tfrac{3}{5}
>				\end{bmatrix}   \\[1.15ex]
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

\[
\boldsymbol{\sigma} = \begin{bmatrix}
	4 & 2 & 0 \\
	2 & 5 & 1 \\
	0 & 1 & 3
\end{bmatrix}
\]

The stress tensor (a fancy way of saying matrix in most situations) tells us about how the materials is compressed/stretched or is being twisted. 

a) Calculate the Determinant of $\mathbf{\sigma}$. (The determinant of the stress tensor provides insight into the state of stress in the material and can be used to assess stability.). 
b) alculate the Trace of $\mathbf{\sigma}$. (The trace of the stress tensor represents the sum of the normal stresses acting on the material, which can influence material behavior under load.)  
c) Calculate the Transpose of $\mathbf{\sigma}$. (The transpose of the stress tensor helps verify its properties, particularly since the stress tensor must be symmetric for physical applications. Is the given stress tensor symmetric? How do you know?)  
d) Calculate the Inverse of $\mathbf{\sigma}$. (The inverse of the stress tensor can be useful for transformations in continuum mechanics, especially in determining the relationship between stress and strain.)  





