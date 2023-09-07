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

[Symbolica](https://symbolica.io) is a computer algebra system which aims to handle expressions with billions
of terms, taking up terabytes of diskspace. It can easily be incorporated into existing projects using its Python, Rust or C++ bindings.

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

## Rust

If you are using Symbolica as a library in Rust, simply include it in the `Cargo.toml`:

```toml
[dependencies]
symbolica = "0.1"
```

## Python

Symbolica can be installed for Python >3.5 using `pip`:

```sh
pip install symbolica
```

The installation may take some time, as it may have to compile Symbolica.

# Examples

In the following example we create a Symbolica expression `(1+x)^2`, expand it, and replace `x^2` by `6`:

```python
from symbolica import Expression
x = Expression.var('x')
e = (1+x)**2
r = e.expand().replace_all(x**2, 6)
print(r)
```
which yields `2*x+7`.

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