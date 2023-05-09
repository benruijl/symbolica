<h1 align="center">
  <br>
  <img src="https://symbolica.io/logo.svg" alt="logo" width="200">
  <br>
</h1>

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://reform.zulipchat.com"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/benruijl/symbolica"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
</p>

# Symbolica

[Symbolica](https://symbolica.io) is a symbolic manipulation toolkit which aims to handle expressions with billions
of terms, taking up terabytes of diskspace. It can easily be incorporated into existing projects using its Python, Rust or C++ bindings.

# Installation

Symbolica is in early development, but can already be used as a library in Rust and Pyton.

## Rust

If you are using Symbolica as a library in Rust, simply include it in the `Cargo.toml`:

```toml
[dependencies]
symbolica = { git = "https://github.com/benruijl/symbolica.git" }
```

## Python

Compile Symbolica with a recent Rust compiler:

```sh
git clone https://github.com/benruijl/symbolica.git
cd symbolica
cargo build --release
```
and copy the shared library to your destination location, stripping the leading `lib` from the filename:
```sh
cp target/release/libsymbolica.so symbolica.so
```

Now test that it works with the following Python script, which showcases pattern matching and rational polynomial arithmetic:

```python
from symbolica import Expression

# create a Symbolica expression
x, y, w1_, w2_ = Expression.vars('x','y','w1_','w2_')
f = Expression.fun('f')
e = f(3,x)*y**2+5

# replace f(w1_,w2_) with f(w-1,w2^2) in e where w1_>=0 and w2_ is a variable
r = e.replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
print('Replaced:', r)

# parse rational polynomials
p = Expression.parse('(x*y^2*5+5)^2/(2*x+5)+(x+4)/(6*x^2+1)').to_rational_polynomial()
print('Rational polynomial:', p)

# compute polynomial GCDs
a = Expression.parse('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
b = Expression.parse('(1 - 3*x1 - 5*x2 - 7*x3 + 9*x4 - 11*x5 - 13*x6 + 15*x7)^2 + 1').to_rational_polynomial()
g = Expression.parse('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 - 15*x7)^2 + 3').to_rational_polynomial()
ag = a * g
bg = b * g
print('Complicated GCD:', ag.gcd(bg))
```

## Development

Symbolica is in early development. Follow the development on [Zulip](https://reform.zulipchat.com)!