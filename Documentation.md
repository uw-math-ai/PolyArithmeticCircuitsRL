# Documentation

I handwrote this (no AI, nothing)! Notice the lack of emojis?

### Polynomial Representation Internally

Polynomial's are internally represented as an array of coefficients. Each entry in the array represents the possible monomial Now a polynomial is the sum of a bunch of monomials. A monimal looks like this:

$$x_0^{e_0} x_1^{e_1} \dots x_{n-1}^{e_{n-1}}$$

Given the exponents $(e_0, e_1, \dots, e_{n-1})$ of this monomial, we can find the position of this monomial in the array using the formula:

$$
\text{index}(e_0, e_1, \dots, e_{n-1}) = \sum_{i=0}^{n-1} e_i \cdot (d+1)^i
$$


Where
* $n$ is the number of variables (ie: $x_0, x_1, ..., x_n$)
* $d$ is the degree

#### Example:

Consider we have a polynomial with $n = 2$ and $d = 2$. Then the array looks like:

$[ \text{coeff}(1), \text{coeff}(x), \text{coeff}(x^2), \text{coeff}(y), \text{coeff}(xy), 0, \text{coeff}(y^2) ]$

Now let's find $xy$:

$\text{index}(xy) = 1 (3 ^ 0) + 1(3^1) = 4$

This lines up!

### Reward Scheme

The reward scheme is the difference between the current vector and the target.