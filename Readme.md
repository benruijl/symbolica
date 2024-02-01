<h1 align="center">
  <br>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://symbolica.io/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://symbolica.io/logo.svg">
  <img src="https://symbolica.io/logo.svg" alt="logo" width="200">
</picture>
  <br>
</h1>

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://reform.zulipchat.com"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/benruijl/symbolica"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
</p>

# Symbolica

Symbolica is a blazing fast and modern computer algebra system which aims to handle huge expressions. It can easily be incorporated into existing projects using its Python, Rust or C++ bindings.
Check out the live [Jupyter Notebook demo](https://colab.research.google.com/drive/1VAtND2kddgBwNt1Tjsai8vnbVIbgg-7D?usp=sharing)!

For documentation and more, see [symbolica.io](https://symbolica.io).



## Quick Example

Symbolica allows you to build and manipulate mathematical expressions through matching and replacing patterns, similar to `regex` for text:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://symbolica.io/resources/demo.dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="https://symbolica.io/resources/demo.light.gif">
  <img width="600" alt="A demo of Symbolica" srcset="https://symbolica.io/resources/demo.dark.gif">
</picture>

You are able to perform these operations from the comfort of a programming language that you (probably) already know, by using Symbolica's bindings to Python, Rust and C++:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://symbolica.io/resources/completion.png">
  <source media="(prefers-color-scheme: light)" srcset="https://symbolica.io/resources/completion_light.png">
  <img width="600" alt="A demo of Symbolica" src="https://symbolica.io/resources/completion.png">
</picture>

# Installation

Visit the [Get Started](https://symbolica.io/docs/get_started.html) page for detailed installation instructions.

## Python

Symbolica can be installed for Python >3.5 using `pip`:

```sh
pip install symbolica
```

The installation may take some time on Mac OS and Windows, as it may have to compile Symbolica.

## Rust

If you want to use Symbolica as a library in Rust, simply include it in the `Cargo.toml`:

```toml
[dependencies]
symbolica = "0.2"
```

# Examples

Below we list some examples of the features of Symbolica. Check the [guide](https://symbolica.io/docs/) for a complete overview.

### Pattern matching

Variables ending with a `_` are wildcards that match to any subexpression.
In the following example we try to match the pattern `f(w1_,w2_)`:

```python
from symbolica import Expression
x, y, w1_, w2_ = Expression.vars('x','y','w1_','w2_')
f = Expression.fun('f')
e = f(3,x)*y**2+5
r = e.replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2))
print(r)
```
which yields `y^2*f(2,x^2)+5`.

### Solving a linear system

Solve a linear system in `x` and `y` with a parameter `c`:

```python
from symbolica import Expression

x, y, c = Expression.vars('x', 'y', 'c')
f = Expression.fun('f')

x_r, y_r = Expression.solve_linear_system(
    [f(c)*x + y + c, y + c**2], [x, y])
print('x =', x_r, ', y =', y_r)
```
which yields `x = (-c+c^2)*f(c)^-1` and `y = -c^2`.

### Series expansion

Perform the Taylor series in `x` of an expression that contains a user-defined function `f`:

```python
from symbolica import Expression

x, y = Expression.vars('x', 'y')
f = Expression.fun('f')

e = 2* x**2 * y + f(x)
e = e.taylor_series(x, 0, 2)

print(e)
```
which yields `f(0)+x*der(1,f(0))+1/2*x^2*(4*y+der(2,f(0)))`.

### Rational arithmetic

Symbolica is world-class in rational arithmetic, outperforming Mathematica, Maple, Form, Fermat, and other computer algebra packages. Simply convert an expression to a rational polynomial:
```python
from symbolica import Expression
x, y = Expression.vars('x','y')
p = Expression.parse('(x*y^2*5+5)^2/(2*x+5)+(x+4)/(6*x^2+1)').to_rational_polynomial()
print(p)
```
which yields `(45+13*x+50*x*y^2+152*x^2+25*x^2*y^4+300*x^3*y^2+150*x^4*y^4)/(5+2*x+30*x^2+12*x^3)`.

## Development

Follow the development and discussions on [Zulip](https://reform.zulipchat.com)!