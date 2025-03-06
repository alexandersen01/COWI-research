## How to use:

### TODO: Convert circle radius into a function of gradient function (cutoff point)

### TODO: introduce symmetry:
- We should try to introduce symmetry by adding a constraint that implies $\text{all circles must have equal angles and distances to each other}$.

This program uses the `pulp` package – which is open source😁

It will figure out the minimum amount of circles/lights (mentioned in the objective function) needed to illuminate a room, and where to place them.

Why doesn't it just place 1 circle and call it a day?
- A constraint listed here is that it needs to have a $L_{\min}$ amount of average coverage of light in one cell to prevent this. The user may define the value of this variable.

The user can adjust the following params: 
- Room dimensions
- Average grid area coverage (in pct, will change to lux when applicable)
- Grid size
- Circle radius

Note:
- $c_{ij, kl}$ is calculated using the gradient function: $\frac{1}{1 + 2d/r}$ where $d$ is the distance from circle center to cell center and $r$ is the circle radius. This can of course (and will) be tweaked in later stages.
- The model assumes that coverage values from multiple circles add linearly at each area cell (this is probably true for irl applications)

### The maths behind it:

**Sets and indices:**

$$
\begin{align}
i, j: \text{ indices for grid points where circles can be placed} \\
k, l: \text{ indices for small area cells (subcells)} \\
G: \text{ set of all valid grid points} \\
N_{ij}: \text{ set of area cells that can be covered by a circle at grid point } (i,j)
\end{align}
$$

**Decision variables:**

$$
\begin{align}
x_{ij} = \begin{cases}
1 \quad\text{if a circle is placed at grid point } {i,j}\\
0 \quad\text{otherwise}
\end{cases}\\
y_{kl} = \text{coverage value (lux) at area cell } (k,l)
\end{align}
$$

**Params:**

$$
\begin{align}
c_{ij, kl} = \text{coverage contribution of a circle at $(i, j)$ to area cell } (k, l) \\
L_{min} = \text{minimum required average light level for each grid cell} \\
M_{ij} = \text{set of area cells in grid cell $(i, j)$ (for averaging constraint)}
\end{align}
$$

**Objective function:**

$$
\begin{align}
\min \sum_{(i, j) \in G} x_{ij}
\end{align}
$$

s.t.

$(11)$ provides coverage calculation for each area cell:

$$
\begin{align}
y_{kl} = \sum_{(i, j) \in G} c_{ij, kl} x_{ij} \quad\forall (k, j) \in A
\end{align}
$$

$(12)$ provides a minimum average light level in each grid cell, where $L_{\min}$ is the given level:

$$
\begin{align}
\frac{\sum_{(k,l) \in M_{ij}} y_{kl}}{|M_{ij}|} \geq L_{min} \quad \forall (i,j) \in G
\end{align}
$$

$(13)$ and $(14)$ provides non-negativity and binary constraints:

$$
\begin{align}
x_{ij} &\in \set{0, 1} \quad\forall (i,j) \in G \\
y_{kl} &\geq 0 \quad\forall (k,l) \in A
\end{align}
$$