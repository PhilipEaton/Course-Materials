---
layout: default
title: Mathematical Methods - Lecture 01
course_home: /courses/math-methods/
nav_section: lectures
nav_order: 1
---

# Lecture 01 -- The Essentials of Linear Algebra


Linear algebra is a powerful tool in physics, widely used for modeling physical phenomena and solving mathematical systems. These applications range from analyzing interactions in particle systems (relevant in solid-state physics and discrete systems) to describing transition probabilities between quantum energy levels, as well as determining transition energies in quantum phenomena.

One of the simplest yet most useful applications of linear algebra is solving systems of equations, which often represent models with constraints or relationships between physical quantities. Linear algebra provides an efficient method for organizing and solving these systems.

At the core of linear algebra are **vectors** and **matrices**, along with their representations:

- **Vectors**, as you may recall from introductory physics, represent quantities with both **magnitude and direction**, such as displacement, velocity, momentum, or force. In physics, vectors may represent either physical quantities or abstract quantities, and they obey rules of vector addition and scalar multiplication, as we will see later in this course.
- **Matrices** serve as a more abstract yet crucial concept in physics. In their simplest form, matrices are rectangular arrays of numbers that operate on vectors. These operations often represent transformations of vectors, such as rotations, scaling (stretches), reflections (parity flips), and other similar effects.


## Vectors

As mentioned above, vectors are sets of numbers that represent both the magnitude and direction of a physical quantity. For example, if we define an origin and a three-dimensional coordinate system, we can specify the position of a particle as being located 3 meters to the right, 1 meter forward, and 4 meters down. Writing this in words can be cumbersome, so we represent vectors in a more concise mathematical form. This can be done in two ways:

$$
\vec{r} = +3\,\hat{i} + 1\,\hat{j} - 4 \,\hat{k} \hspace{1cm} \text{or} \hspace{1cm} \vec{r} = \begin{bmatrix} +3 \\ +1 \\ -4 \end{bmatrix} 
$$

The left-hand representation uses **unit basis vectors** $(\hat{i}, \hat{j}, \hat{k})$, while the right-hand representation uses a **column vector** format.

Now, **unit basis vectors** are vectors with a **magnitude of 1 that point in independent directions**. This means that each unit vector has a length of 1 (they have no units) and all point in different directions when plotted:

<img
  src="{{ '/courses/math-methods/images/lec01/basisvectors.svg' | relative_url }}"
  alt="3D coordinate axes showing unit vectors i-hat, j-hat, and k-hat."
  style="display:block; margin:1.5rem auto; max-width:600px; width:50%;">

where the dashed arrows represent the corrdinate system and the solid arrows are the unit vectors. Notice the unit vector $\hat{i}$  points along the $x$-axis specitically and since it is a unit vector the magnitude of $\hat{i}$ can be written as:

$$ \text{Magnitude}(\hat{i}) = |\hat{i}| = 1 \hspace{1cm}$$ 

Similarly for the other unit vectors:

$$|\hat{j}| = 1 \hspace{1cm} |\hat{k}| = 1$$

and $\hat{j}$ points along the $y$-axis and $\hat{k}$ points along the $z$-axis.

The basis set used in the above example not only point in indedependent directions, but point in orthogonal (perpendicular) directions. This makes $(\hat{i}, \hat{j}, \hat{k})$ mutually orthogonal, unit basis vectors, which have very nice properties when we get to multiplying vectors together. We will explore this concept further in Lecture 04 when we discuss Vector Operations, but for now, a general pictorial understanding will suffice.

An interesting observation arises from the two different representations of the position vector given above. Since both representations describe the exact same position vector, we should be able to write:

$$\begin{aligned}
	+3\,\hat{i} + 1\,\hat{j} - 4 \,\hat{k} &= \begin{bmatrix} +3 \\ +1 \\ -4 \end{bmatrix} \\
	&= \begin{bmatrix} +3 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ +1 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ -4 \end{bmatrix} \\
	+3\,\hat{i} + 1\,\hat{j} - 4 \,\hat{k} &= +3\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} + 1 \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} -4 \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} 
\end{aligned}$$

and so we have a link between the two representations with:

$$ \hat{i} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} \hspace{2cm} \hat{j} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \hspace{2cm} \hat{k} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}   $$

There are multiple ways to represent the basis unit vectors, meaning there is no single unique representation to describe positions in 3D space. However, the approach we used is certainly the simplest. 


---

### A Quick Detour: What is a Basis, Really?

The three vectors $\hat{i}$, $\hat{j}$, and $\hat{k}$ are what we call a **basis** for 3D space. A **basis** is just a set of vectors that can be used to build any other vector in that space using addition and scalar multiplication.

This ability to combine a set of vectors to recreate any vector in the space is called **spanning**. More formally, we say the vectors **span the space**. For example, in 3D, this looks like:

$$
\vec{v} = a \, \hat{i} + b \, \hat{j} + c \, \hat{k}
$$

where $a$, $b$, and $c$ are scalars. It’s easy to see that any vector in 3D space can be written this way, so $\hat{i}$, $\hat{j}$, and $\hat{k}$ together form a basis.

There’s one more key property a set of basis vectors need to have: the basis vectors must also be  **linearly independent**. That means you can’t build one of them using a combination of the others. Each one brings something fundamentally new; they all point in different directions, and none of them are redundant.

Here’s a helpful way to think about it: imagine a big box of LEGO bricks. Suppose the set has only three distinct pieces; say, in different shapes and/or colors. If you can build *anything* you want by snapping together just those three kinds of pieces in different ways, then those three form a basis. You don’t need any more, and you can’t get away with fewer.

In our 3D space, the standard basis vectors:

$$
\hat{i} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} \qquad \hat{j} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \qquad \hat{k} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

are three fundamental "LEGO bricks", each being distinct in that they each point in different directions. Any vector in 3D space can be expressed as a combination of these—like writing a recipe for a location: "go 3 units in the \\(\hat{i}\\) direction, then 1 unit in the \\(\hat{j}\\) direction, and finally $-$4 units in the \\(\hat{k}\\) direction."


{% include result.html content="
A set of basis vectors must satisfy two conditions:

1) the vectors must span the space (i.e., any vector in the space can be created using only the basis vectors), and  
2) the vectors must be linearly independent (none of them is a redundant combination of the others).
" %}

It turns out there are many possible bases for 3D space, not just the basis discussed previously. For example, we could rotate the proposed basis vectors, stretch them, or even choose weirdly slanted directions, and as long as the vectors are still linearly independent and span the space, they still form a valid basis. But the *standard basis* with its clean, perpendicular directions is by far the most convenient.

---

Now, that the column vectors, as we have written them, for a basis is the result of a few assumptions that we should address, particularly regarding how column vectors are added and how constants are multiplied to them. The methods we used above are so intuitive that many readers may not have realized we performed an operation that we had not explicitly defined as allowed. Specifically, when adding column vectors, you combine the corresponding elements from each vector, and when multiplying by a constant (i.e., a scalar), you multiply every element of the vector by that constant.


{% include example.html content="
Let's consider adding the following matrices together:

$$ \begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} \qquad \qquad \begin{bmatrix} -3 \\ 5 \\ 0 \end{bmatrix}$$

Adding together, element by element, gives:

$$ \begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} + \begin{bmatrix} -3 \\ 5 \\ 0 \end{bmatrix} = \begin{bmatrix} (+1) + (-3) \\ (-4) + (+5) \\ (+2) + (0) \end{bmatrix} = \begin{bmatrix} -2 \\ +1 \\ +2 \end{bmatrix} $$

We could also ask what happens when we multiply a vector by a constant. When you multiply a column vector by a constant, you multiply each element of the vector by that constant. For example:

$$ 3 \begin{bmatrix} 3 \\ -4 \\ -1 \end{bmatrix} = \begin{bmatrix} 3(+3) \\ 3(-4) \\ 3(-1) \end{bmatrix} = \begin{bmatrix} +9 \\ -12 \\ -3 \end{bmatrix} $$

and similarly for negative numbers. 

In fact, this is how subtraction can be defined:

$$\begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} - \begin{bmatrix} -3 \\ 5 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} + (-1) \begin{bmatrix} -3 \\ 5 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} + \begin{bmatrix} +3 \\ -5 \\ 0 \end{bmatrix} = \begin{bmatrix} 4 \\ -9 \\ 2 \end{bmatrix}  $$

$$\begin{bmatrix} (+1) - (-3) \\ (-4) - (+5) \\ (+2) - (0) \end{bmatrix} \phantom{= \begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} + (-1) \begin{bmatrix} -3 \\ 5 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} + \begin{bmatrix} +3 \\ -5 \\ 0 \end{bmatrix} = \begin{bmatrix} 4 \\ -9 \\ 2 \end{bmatrix} } $$

$$\begin{bmatrix} +4 \\ -9 \\ +2 \end{bmatrix} \phantom{= \begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} + (-1) \begin{bmatrix} -3 \\ 5 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ -4 \\ 2 \end{bmatrix} + \begin{bmatrix} +3 \\ -5 \\ 0 \end{bmatrix} = \begin{bmatrix} +4 \\ -9 \\ +2 \end{bmatrix} } $$

Notice, directly subtracting the elements, the vertical steps, and treating subtraction as addition combined with the multiplication by -1 both give the same result. 
" %}


We will talk more about vectors and their various operations in Lecture 04. 






## Matrices

In the most general terms, matrices are a rectangular configuration of numbers that can mean pretty much anything you want:

$$ 
\text{Matrix named } A = \underline{\underline{A}} =  \mathbf{A} =  \begin{bmatrix} +4 & - 2 & 7  \\ -9 & 0 & -4 \\ +5 & -5 & 2 \end{bmatrix}  
$$

When writing by hand, the double underlined notation is often the easiest notation to use when indicating a variable is a matrix. In text, however, the bold-faced notation is most commonly used. 

In physics matrices are used for a wide variety of reasons, but the most common of which is to represent some form of coordinate transformation or a transition in a system of some kind. Throughout out this unit on linear algebra we will see that matrices can be used to rotate, rescale, and  flip vectors. Let's looks into the key features of a matrix and then jump into one of the simplest ways we can use matrices -- solving systems of linear equations.

### Anatomy of a Matrix

Let's cover the basic structure of a matrix. Just like we learned how to identify components of a vector—like its direction and magnitude—we should take a moment to understand the parts of a matrix. This will help us communicate clearly and make sure we’re all speaking the same mathematical language.


- A **matrix** is a rectangular array of numbers arranged in **rows** (horizontal) and **columns** (vertical).
- The **shape** of a matrix tell us how many rows and how many columns it has. A matrix with $ m $ rows and $ n $ columns is called an $ m \times n $ matrix (read “$ m $ by $ n $”). Rows come first, then columns—like writing an address: “row, column.” Remember, *“Rail Car”*.
- Each number inside the matrix is called an **element**. The entry located in the $ i $-th row and $ j $-th column is labeled as $ a_{ij} $, where the first subscript points to the row and the second to the column.
- The **main diagonal** of a matrix begins with the top-left element and continues down and to the right, one down and one to the right, and etc. 
- The elements $ a_{11}, a_{22}, a_{33} $ lie on the **main diagonal**, which stretches from the top left to the bottom right of the matrix:

	$$
	\text{Main diagonal: } a_{11} \rightarrow a_{22} \rightarrow a_{33} \rightarrow \cdots \rightarrow a_{nn}
	$$	

- If all entries are zero, we call it a **zero matrix**. If all main diagonal entries are 1 and everything else is 0, we call it the **identity matrix** (we’ll meet this matrix soon).
	

{% include example.html content="
For example, consider the matrix below:

$$
\mathbf{A} = \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6
\end{bmatrix}
$$

Let’s break this down:  

- This matrix has 2 rows and 3 columns, so it is a $ 2 \times 3 $ matrix.
- The entry in the first row, second column is $ a_{12} = 2 $.
- The entry in the second row, third column is $ a_{23} = 6 $.
- Main diagonal: $ 1 \rightarrow 5 $.
" %}

Later on, when we perform operations like multiplying matrices or solving systems of equations, keeping track of rows and columns will be crucial. So make sure you’re comfortable with this anatomy since it’ll save you a lot of confusion later!

Now that we’ve gotten to know the basic structure of a matrix, let’s start doing something with them! One of the simplest and most intuitive operations we can perform is combining matrices through addition and subtraction.


### Matrix Addition and Subtraction

**Two matrices can be added or subtracted only if they have the same shape**. Given two matrices \\(\mathbf{A}\\) and \\(\mathbf{B}\\) of size \\(m \times n\\), their sum \\(\mathbf{C} = \mathbf{A} + \mathbf{B}\\) and their difference \\(\mathbf{D} = \mathbf{A} - \mathbf{B}\\) are computed in a manner similar to vector addition and subtraction. 

{% include example.html content="
For example, if we have the matrices

$$
\mathbf{A} = \begin{bmatrix}
    1 & 2 \\
    3 & 4
\end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix}
    5 & 6 \\
    7 & 8
\end{bmatrix},
$$

then their sum and difference are calculated as follows:

$$
\mathbf{C} = \mathbf{A} + \mathbf{B} = \begin{bmatrix}
    1 + 5 & 2 + 6 \\
    3 + 7 & 4 + 8
\end{bmatrix} = \begin{bmatrix}
    6 & 8 \\
    10 & 12
\end{bmatrix}
$$

$$
\mathbf{D} = \mathbf{A} - \mathbf{B} = \begin{bmatrix}
    1 - 5 & 2 - 6 \\
    3 - 7 & 4 - 8
\end{bmatrix} = \begin{bmatrix}
    -4 & -4 \\
    -4 & -4
\end{bmatrix}
$$
" 
%}




### System of Linear Equations

Recall from algebra that a system of linear equations is a set of $n$ linear equations of $m$ variables. For example, the following is a system on linear equations with two equations $n=2$ and three variables $m = 3$:

$$
\begin{aligned} 2x + 3y - 3 z &= 6 \\ 
4x - y + 4 z &= 5
\end{aligned}
$$

This system of equations can be written as a matrix equation in the following manner:

$$\underbrace{\begin{bmatrix} 2 & 3 & -3 \\ 4 & -1 & 4 \end{bmatrix}}_\text{Let's call this $\mathbf{A}$} \underbrace{\begin{bmatrix} x \\ y \\ z\end{bmatrix}}_{\vec{r}} = \underbrace{\begin{bmatrix} 6 \\ 5\end{bmatrix}}_{\vec{b}} \quad\implies\quad \mathbf{A} \vec{r} = \vec{b}
$$

where the matrix \\( \mathbf{A} \\) is called the **coefficient matrix**, \\( \vec{r} \\) is the **variable vector**, and \\( \vec{b} \\) represents the **results vector**. 

At this point, it's important to discuss how matrices are generally described. The dimensions of a matrix are indicated by the number of rows and columns, expressed as (number of rows) \\(\times\\) (number of columns). For example, the matrix \\( \mathbf{A} \\) is a \\( 2 \times 3 \\) matrix (read as ``2 by 3") because it has 2 rows and 3 columns. The column vector \\( \vec{r} \\) is a \\( 3 \times 1 \\) matrix, while the results vector \\( \vec{b} \\) is a \\( 2 \times 1 \\) matrix. 

Notice what happened here: we took a \\( 2 \times 3 \\) matrix, acted on a \\( 3 \times 1 \\) vector, and got a \\( 2 \times 1 \\) vector out as a result. This is sometimes called the ``rows into columns'' rule, where the rows of the second object, the \\( 3 \times 1 \\) vector in this case, must be the same number as the columns in the first, the \\( 2 \times 3 \\) matrix $\mathbf{A}$. 

$$
\overset{\text{2}\times\textbf{3}}{\begin{bmatrix}
2 & 3 & -3 \\
4 & -1 & 4
\end{bmatrix}}
\;
\overset{\textbf{3}\times\text{1}}{\begin{bmatrix}
x \\ y \\ z
\end{bmatrix}}
=
\overset{\text{2}\times\text{1}}{\begin{bmatrix}
6 \\ 5
\end{bmatrix}}
$$

The resulting object will have the same number of rows as the first object, 2 rows from matrix $\mathbf{A}$, and the same number of columns as the second, 1 column from vector $\vec{r}$. This means, in this case, the resulting object, $\vec{b}$, should be a \\( 2 \times 1 \\) vector, which it is!

Another takeaway here is that all vectors are matrices with either a single row or a single column and matrices can take on any shape. A vector represented by a single row is called a **row vector**, and when represented by a single column, it is called a **column vector**.

This also means the two operations on vectors we discussed above are similarly true for matrices. When adding matrices you add elements of identical positions in their matrices together. For example,

$$ 
\begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} + \begin{bmatrix} -1 & 0 \\ 6 & -9 \end{bmatrix} = \begin{bmatrix} (+2) + (-1) & (+3) + (0) \\ (+4) + (+6) & (+5) + (-9) \end{bmatrix} = \begin{bmatrix} +1 & +3 \\ +10 & -4 \end{bmatrix}
$$

Notice that with this definition of matrix addition (and similarly, matrix subtraction), **you can only add or subtract matrices of the same size**! In the example above, we added two $2 \times 2$ matrices and obtained a single $2 \times 2$ matrix as the result. **You cannot add or subtract matrices of different sizes!**

Fortunately, multiplying a matrix by a scalar doesn’t depend on the matrix’s size—you simply multiply every element of the matrix by that scalar:

$$ -3 \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} = \begin{bmatrix} -3(+2) & -3(+3) \\ -3(+4) & -3(+5) \end{bmatrix} = \begin{bmatrix} -6 & -9 \\ -12 & -15 \end{bmatrix}$$

Now, let’s revisit the matrix equation from earlier. We need to define another matrix operation: how a matrix operates on a vector (or even another matrix). This process is summarized by the statement: **rows into columns**.

To see this in action, consider the following example:

$$
\begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix} = ? 
$$

To carry out this multiplication, we multiply each element in the row vector on the left by the corresponding element in the column vector on the right. Specifically, the first element of the row multiplies the first element of the column, the second element multiplies the second, and so on. Afterward, we sum the products:

$$ 
\begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix} = (1)(3) + (2)(4) = 3 + 8 = 11  
$$

Notice that this result is a single number (or scalar), rather than a matrix or vector. Why? This follows from the “rows into columns” rule: when you multiply a $1 \times 2$ matrix by a $2 \times 1$ matrix, you get a $1 \times 1$ matrix, or a scalar.

This example highlights some key rules of **matrix multiplication**:

1. Matrices can only be multiplied if the number of columns in the first matrix matches the number of rows in the second matrix. In other words, a matrix of size $n \times m$ can multiply a matrix of size $m \times p$ (in that order). However, you cannot reverse the order and multiply a $m \times p$ matrix by an $n \times m$ matrix.
	
>	Matrix multiplication is not commutative — the order of multiplication matters!
	
2. The size of the resulting matrix can be determined by deleting the column count of the first matrix and the row count of the second matrix, using only the remaining row count of the first and the column count of the second. For example, when multiplying a $n \times m$ matrix by an $m \times p$ matrix, the result will be a matrix of size $n \times p$.



{% include example.html content="
Let's look at an example to illustrate this, using our matrix equation from earlier:

$$ \mathbf{A} \vec{r} = \vec{b} \implies \begin{bmatrix} 2 & 3 & -3 \\ 4 & -1 & 4 \end{bmatrix} \begin{bmatrix} x \\ y \\ z\end{bmatrix} = \begin{bmatrix} 6 \\ 5\end{bmatrix} $$

Here, we have a $2 \times 3$ matrix multiplying a $3 \times 1$ matrix, so this multiplication is allowed. The result will be a $2 \times 1$ matrix (2 rows and 1 column).

To compute this, we take each row of the left matrix and multiply it by the single column of the right matrix:

- The first row with the first column gives:

$$ 2x + 3y - 3z $$

- The second row with the first column gives:

$$ 4x - y + 4z $$

Combining these into a $2 \times 1$ matrix, we rewrite the matrix equation as:

$$ \begin{bmatrix} 2x + 3y - 3z \\ 4x - y + 4z \end{bmatrix} = \begin{bmatrix} 6 \\ 5 \end{bmatrix} $$

Equating elements on both sides, we obtain the system of equations:

$$ \begin{aligned} 
    2x + 3y - 3z &= 6 \\ 
    4x - y + 4z &= 5 
\end{aligned} $$

which is exactly the same system of linear equations we started with. This confirms that everything is consistent.
" %}


{% include example.html content="
One more example should serve to put matrix multiplication into full working order. Consider:

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \quad \text{and} \quad \mathbf{B} = \begin{bmatrix} 7 & 8 \\ 9 & 10 \end{bmatrix}
$$

To calculate \\( \mathbf{A} \mathbf{B} \\), we multiply each row of \\( \mathbf{A} \\) by each column of \\( \mathbf{B} \\), resulting in a \\( 3 \times 2 \\) matrix.

$$
\mathbf{A} \mathbf{B} = \begin{bmatrix} 
    (1 \times 7) + (2 \times 9) & (1 \times 8) + (2 \times 10) \\ 
    (3 \times 7) + (4 \times 9) & (3 \times 8) + (4 \times 10) \\ 
    (5 \times 7) + (6 \times 9) & (5 \times 8) + (6 \times 10) 
\end{bmatrix}
= \begin{bmatrix} 
    25 & 28 \\ 
    57 & 64 \\ 
    89 & 100 
\end{bmatrix}
$$
" %}

Notice, in the previous example, that the first row, \\(\begin{bmatrix} 1 & 2 \end{bmatrix}\\), multiplied by the first column, \\(\begin{bmatrix} 7 & 9 \end{bmatrix}\\), results in the element \\(25\\) in the first row and first column of the resulting matrix. Similarly, the second row, \\(\begin{bmatrix} 3 & 4 \end{bmatrix}\\), multiplied by the second column, \\(\begin{bmatrix} 8 & 10 \end{bmatrix}\\), produces the element \\(64\\) in the second row and second column of the resulting matrix. This is the procedure in matrix multiplication:


<div class="result">
The i-th row and j-th column element of the resulting matrix is obtained by multiplying the i-th row of the first matrix with the j-th column of the second matrix.
</div>



## Matrix Element Notation

To make it easier to refer to specific entries inside a matrix, mathematicians and physicists use a standardized element notation. We define the elements of a matrix as follows:

> The element in the *i*-th row and *j*-th column of a matrix $\mathbf{A}$ is denoted as $a_{ij}$.


\noindent
The subscript \\(ij\\) always follows the pattern: \textit{row, then column}. 

For example, if

$$
\mathbf{A} = \begin{bmatrix} 
	2 & 5 & -1 \\ 
	4 & 0 & 3 
\end{bmatrix}
$$

then:
- $a_{11} = 2$ (first row, first column),
- $a_{12} = 5$ (first row, second column),
- $a_{23} = 3$ (second row, third column).


It’s important to remember:
- Matrix indexing starts at \\(1\\), not \\(0\\). (Computer programmers, take note!)
- Each $a_{ij}$ refers to a **single element** of the matrix, not an entire row or column.

This notation becomes extremely helpful when writing general matrix equations, defining operations like the transpose, or describing algorithms like Gaussian elimination more precisely.







## Solving Systems of Linear Equations:

Let's look at how matrices can be used to solve systems of linear equations. Consider the example we gave above: 

$$\begin{aligned} 2x + 3y - 3 z &= 6 \\ 4x - y + 4 z &= 5\end{aligned}$$

which can be rewritten in the following manner:

$$ \underbrace{\begin{bmatrix} 2 & 3 & -3 \\ 4 & -1 & 4 \end{bmatrix}}_\text{Let's call this $\mathbf{A}$} \underbrace{\begin{bmatrix} x \\ y \\ z\end{bmatrix}}_{\vec{r}} = \underbrace{\begin{bmatrix} 6 \\ 5\end{bmatrix}}_{\vec{b}} \quad\implies\quad \mathbf{A} \vec{r} = \vec{b}
$$

This is one way we could use matrices to rewrite this equation, another way to do it is to construct the **augmented matrix** for this system of linear equations. The augmented matrix is constructed by taking the coefficient matrix $\mathbf{A}$ and tacking the result vector onto the end, in the following manner:

$$
\begin{bmatrix}
	2 & 3 & -3 & \vline & 6 \\
	4 & -1 & 4 & \vline & 5
\end{bmatrix}
$$

where the vertical line inside the matrix is a visual aid indicating there the coefficients end and the answers begin. The vertical line is not an actual mathematical operator, just an aid.


Now we can take advantage of the addition method of solving a system of equations to solve this problem. Recall the addition method is used to eliminate variables between the two equations by multiplying one equation by a constant and then adding it to the other. For example, if we multiply the first equation by $-2$:

$$
-2(2x + 3y - 3 z = 6) \quad \longrightarrow \quad   -4 x - 6 y + 6 z = -12 
$$

and add it to the second equation, we get

$$
\begin{array}{crcl}
	&-4x - 6y + 6z & = & -12 \\
	+ & 4x - y + 4z & = & 5  \\\hline
	&0 - 7 y + 10 z &=& -7
\end{array}
$$

We can then use this equation in place of the first or second equation. Note, this is not a new equation, it replaces one of out old equations. So we are left with:

$$
\begin{aligned} 
2x + 3y - 3 z &= 6 \\ 
- 7 y + 10 z &= -7
\end{aligned}
$$

This would have an augmented matrix of the form:

$$
\begin{bmatrix}
	2 & 3 & -3 & \vline & 6 \\
	0 & -7 & 10 & \vline & -7
\end{bmatrix}
$$

Notice this result is exactly what we would get if we had taken the first row of the original augmented matrix, multiplied it by $-2$, added it to the second row, and then stored the result in the second row. This reveals a useful shortcut: we can solve systems of linear equations using the addition method without needing to repeatedly write down the variables $x$, $y$, and $z$ each time. This approach simplifies the process considerably.

### Gaussian Elimination (Row Reduction) Process

There is a standard process of systematically reducing elements of the augmented matrix to zero. This process is called Gaussian Elimination or, as it is more simply know, Row Reduction. 

This process, as you will see in the following example, may seen extremely specific and not something we would use a lot in the ``real world'' when working on physics problems. This thought is incorrect. This processes is used in many situations, we just do it on computers rather than by hand. Here is a short list of places this processes can be used:

1. **Circuit Analysis (Kirchhoff's Laws)**: In electrical circuits, Kirchhoff's Current Law and Kirchhoff's Voltage Law often lead to systems of linear equations. For example, analyzing current and voltage in circuits with multiple loops and junctions frequently requires solving for unknown currents or voltages using a system of equations.
2. **Forces and Equilibrium (Statics)**: In mechanical systems, especially in engineering, static equilibrium conditions require that the sum of forces and torques on an object be zero. This leads to systems of linear equations involving forces in different directions.
3. **Quantum Mechanics (Matrix Mechanics)**: Quantum states are described by vectors, and observable quantities (such as energy) are represented by operators that act on these states (i.e., matrices). Eigenvalue problems (we will see these later) in quantum mechanics are often solved by setting up and solving systems of linear equations.
4. **Optics (Ray Tracing and Lens Systems)**: In geometrical optics, ray tracing through a system of lenses can lead to a system of linear equations. These equations relate object and image distances with focal lengths and angles.
5. **Vibrations and Normal Modes**: In systems of coupled oscillators (e.g., masses connected by springs), the equations of motion are often a set of coupled linear differential equations. These can be simplified into systems of linear algebraic equations to find normal modes and frequencies. We will look are examples of this at the end of this class.
6. **Thermodynamics and Chemical Equilibrium**: In chemical reactions, conservation laws for mass and charge can lead to a system of linear equations that describes how different species in a reaction are balanced.


{% include example.html content="
Let’s go through an example of this method for the following system of linear equations:

$$
\begin{aligned}
    2x + 3y - z &= 5 \\
    -x + 4y + 2z &= 3 \\
    3x - y + z &= -4
\end{aligned}
\implies 
\begin{bmatrix}
    2 & 3 & -1 & \vline & 5 \\
    -1 & 4 & 2 & \vline & 3 \\
    3 & -1 & 1 & \vline & -4 
\end{bmatrix}
$$

First, lets get the very first element, the $a_{11}$ element, to be 1. This can be done in a couple of ways: i) we could multiply the first row by 1/2, or ii) we could swap row 1 and row 2 (multiplied by -1). Here, to avoid fractions for as long as possible, we will use the second option. This gives:

$$
\begin{bmatrix}
    1 & -4 & -2 & \vline & -3 \\
    2 & 3 & -1 & \vline & 5 \\
    3 & -1 & 1 & \vline & -4 
\end{bmatrix}
$$

Now we can multiply the first row by $-2$, add it to the second row, and store the result in the second row to eliminate the first element of the second row. we can do something similar for the third row --  multiply the first row by $-3$, add it to the third row, and store the result in the third row. These processes give:

$$
\begin{bmatrix}
    1 & -4 & -2 & \vline & -3 \\
    0 & 11 & 3 & \vline & 11 \\
    0 & 11 & 7 & \vline & 5 
\end{bmatrix}
$$

We can eliminate the second term in the last row by multipliying the second row by $-1$, adding it to the third row, and storing the result in the third row:

$$
\begin{bmatrix}
    1 & -4 & -2 & \vline & -3 \\
    0 & 11 & 3 & \vline & 11 \\
    0 & 0 & 4 & \vline & -6 
\end{bmatrix}
$$

Putting this back into a system of linear equations format gives:

$$
\begin{bmatrix}
    1 & -4 & -2 & \vline & -3 \\
    0 & 11 & 3 & \vline & 11 \\
    0 & 0 & 4 & \vline & -6 
\end{bmatrix}
\implies
\begin{aligned}
    x  - 4y - 2z &= -3 \\
    11y + 3z &= 11 \\
    4z &= -6
\end{aligned}
$$

From the third row we can see that $z = -\tfrac{3}{2}$. Putting this into the second equation gives:

$$
\begin{aligned}
    11y + 3z &= 11\\
    11y + 3\left(-\tfrac{3}{2}\right) &= 11  \\
    11y -\tfrac{9}{2} &= 11 \\
    11y  &= \tfrac{31}{2} \\
    y  &= \tfrac{31}{22} 
\end{aligned}
$$

and putting both $z$ and $y$ into the first equation gives:

$$
\begin{aligned}
    x  - 4y - 2z &= -3 \\
    x  - 4\left(\tfrac{31}{22}\right) - 2\left(-\tfrac{3}{2}\right) &= -3 \\
    x  - \tfrac{62}{11} + 3 &= -3 \\
    x  - \tfrac{62}{11} &= -6 \\
    x  &= - \tfrac{4}{11} 
\end{aligned}
$$

So, we have the solution 

$$ x = - \frac{4}{11} \hspace{1cm} y = \frac{31}{22} \hspace{1cm} z = -\frac{3}{2} $$

Depending on what the initial system of equations represented, these results could represent currents throughout a circuit, forces in a static equilibrium problem, and etc.
" %}






## Summary: Core Skills You Should Know

In this lecture, we introduced the foundational tools of linear algebra that we will build on throughout this course: vectors, matrices, and systems of linear equations.

- **Vectors** were reintroduced as mathematical objects that represent both magnitude and direction. We discussed two common representations: basis vector notation (using $\hat{i}$, $\hat{j}$, and $\hat{k}$) and column vector form. We also reviewed basic operations with vectors, including addition, scalar multiplication, and subtraction.
- **Matrices** were introduced as rectangular arrays of numbers that can represent coordinate transformations, system transitions, or collections of coefficients. We reviewed basic matrix operations: addition, scalar multiplication, and most importantly, matrix multiplication. We emphasized the ``rows into columns'' rule for matrix multiplication and highlighted that matrix multiplication is not commutative.
- **Systems of Linear Equations** were shown to have a natural and efficient representation using matrices. We developed the matrix form $\mathbf{A}\vec{r} = \vec{b}$ and learned how to solve systems using Gaussian elimination (also called row reduction), reinforcing the connection between matrix operations and familiar algebraic methods for solving equations.

Throughout the lecture, we stressed the importance of understanding both the operations and the underlying structures. In physics and engineering, vectors and matrices are far more than mathematical formalities—they provide an efficient language for modeling real-world systems, from particle motion to quantum transitions to circuit analysis.

In the coming lectures, we will build on these ideas by introducing more advanced concepts like matrix inverses, determinants, and eigenvalues. These topics will give us powerful tools for solving increasingly complex problems and deepen our understanding of how systems behave.


**Key Takeaway:** Matrices and vectors are the essential ``building blocks'' of the linear world, and mastering them makes solving real problems across physics, engineering, and beyond possible.










## Problems:

- Please keep your work organized and neat.
- Solutions should follow a logical progression and should not skip major conceptual/mathematical steps.
- Provide brief explanations for non-trivial mathematical steps beyond simple algebra.

### Problem 1:

Consider the following matrices:

$$
\mathbf{A} = \begin{bmatrix} 2 & 3 \\ 1 & 4 \\ -1 & 2 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 5 & -2 \\ 3 & 6 \end{bmatrix}
$$

a) Multiply the matrices \\(\mathbf{A}\\) and \\(\mathbf{B}\\) to find the resulting matrix \\(\mathbf{C} = \mathbf{A} \cdot \mathbf{B}\\). If this operation is not allowed, explain why.  

b) If the matrix multiplication is permitted, determine the size of the resulting matrix \\(\mathbf{C}\\). Does this result agree with the rules established earlier?  

c) Multiply the matrices \\(\mathbf{B}\\) and \\(\mathbf{A}\\) to find the resulting matrix \\(\mathbf{D} = \mathbf{B} \cdot \mathbf{A}\\). If this operation is not allowed, explain why.  

d) If the matrix multiplication is permitted, determine the size of the resulting matrix \\(\mathbf{D}\\). Does this result agree with the rules established earlier?  


### Problem 2:

Consider the following circuit with three loops and three resistors. The circuit contains two voltage sources, \\( V_1 = 10 \, \text{V} \\) and \\( V_2 = 5 \, \text{V} \\), and three resistors with values \\( R_1 = 2 \, \Omega \\), \\( R_2 = 3 \, \Omega \\), and \\( R_3 = 4 \, \Omega \\).

Using Kirchhoff's Voltage Law, we obtain the following system of equations for the currents \\( I_1 \\), \\( I_2 \\), and \\( I_3 \\) flowing through each loop:

$$
\begin{aligned}
	2 I_1 + 3 I_2 &= 10, \\
	-2 I_1 + 4 I_3 &= 5, \\
	3 I_2 - 4 I_3 &= -5.
\end{aligned}
$$

a) Write this system of linear equations in matrix form, \\(\mathbf{A} \vec{I} = \vec{V}\\), where \\(\mathbf{A}\\) is the matrix of coefficients, \\(\vec{I}\\) is the vector of unknown currents, and \\(\vec{V}\\) is the vector of voltage values.  

b) Write out the augmented matrix for this system of linear equations. Remember to include 0's if a variable is not present in an equation!	  

c) Solve for the currents \\( I_1 \\), \\( I_2 \\), and \\( I_3 \\) using the Gaussian elimination process described in the Application section.  

d) Interpret your solution: what do the values of \\( I_1 \\), \\( I_2 \\), and \\( I_3 \\) indicate about the direction and magnitude of currents in each loop?  











