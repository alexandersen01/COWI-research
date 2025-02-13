## How to use:

#### TODO: refactor obj func to:

$$
\begin{align}
\max z= \text{area} \geq w_1 \sum_{kl}y_{kl} - w_2\sum_{ij}x_{ij}
\end{align}
$$

This program uses the `pulp` package – which is open source😁

The user can adjust the following params: 
- Room dimensions
- Weight between circles and coverage (which is more important?)
- Grid size
- Circle radius

Higher $\frac{w_1}{w_2}$ ratio $\to$ strongly prioritizes coverage

Lower $\frac{w_1}{w_2}$ ratio $\to$ uses fewer circles but might sacrifice some coverage

### The maths behind it:

Objective function: 

$$
\begin{align}
\max z=w_1 \sum_{kl}y_{kl} - w_2\sum_{ij}x_{ij}
\end{align}
$$

Where: 
- $w_1$ is the area coverage weight
- $w_2$ is the circle weight

Together they maximize covered area while they minimise the amount of circles used

s.t.

$$
\begin{align}
y_{kl} \leq \sum_{ij}a_{ijkl}x_{ij}\quad\forall k, l
\end{align}
$$

Binary variables: 

$$
x_{ij}=
\begin{cases}
1\quad\text{if a circle is placed at grid point }(x,y)\\
0\quad\text{otherwise}
\end{cases}
$$

$$
y_{kl}=
\begin{cases}
1\quad\text{if a cell at $(k,l)$ is covered}\\
0\quad\text{otherwise}
\end{cases}
$$

$$
a_{ijkl} = 
\begin{cases}
1\quad\text{if a circle at ($ij$) can cover cell ($kj$)}\\
1\quad\text{if }(i-k)^2 + (j-l)^2 \leq (r + \frac{s}{2})^2\\
0\quad\text{otherwise}
\end{cases}
$$


This ensures that a cell can only be marked as covered if at least one circle covers it ($y_{kl} = 1$)