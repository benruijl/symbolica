use std::sync::Arc;

use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{finite_field::Zp, integer::Z},
    poly::{factor::Factorize, polynomial::MultivariatePolynomial, Variable},
};

fn factor_ff_univariate() {
    let exp = Atom::parse("x^100-1").unwrap().expand();

    let field = Zp::new(17);
    let poly: MultivariatePolynomial<_, u8> = exp.to_polynomial(&field, None);

    let factors = poly.square_free_factorization();

    println!("Factorization of {}:", poly);
    for (f, pow) in factors {
        println!("\t({})^{}", f, pow);
    }
}

fn factor_ff_bivariate() {
    let order = Arc::new(vec![
        Variable::Symbol(Symbol::new("x")),
        Variable::Symbol(Symbol::new("y")),
    ]);

    let input = "((y+1)*x^2+x*y+1)*((y^2+2)*x^2+y+1)";

    let exp = Atom::parse(input).unwrap().expand();

    let field = Zp::new(17);
    let poly: MultivariatePolynomial<Zp, u8> = exp.to_polynomial(&field, Some(order));

    println!("Factorization of {}:", poly);
    for (f, pow) in poly.factor() {
        println!("\t({})^{}", f, pow);
    }
}

fn factor_ff_square_free() {
    let exp = Atom::parse("(1+x)*(1+x^2)^2*(x^4+1)^3").unwrap().expand();

    let field = Zp::new(3);
    let poly: MultivariatePolynomial<_, u8> = exp.to_polynomial(&field, None);

    let factors = poly.square_free_factorization();

    println!("Square-free factorization of {}:", poly);
    for (f, pow) in factors {
        println!("\t({})^{}", f, pow);
    }
}

fn factor_square_free() {
    let exp = Atom::parse("3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)")
        .unwrap()
        .expand();

    let poly: MultivariatePolynomial<_, u8> = exp.to_polynomial(&Z, None);

    let factors = poly.square_free_factorization();

    println!("Square-free factorization of {}:", poly);
    for (f, pow) in factors {
        println!("\t({})^{}", f, pow);
    }
}

fn factor_univariate_1() {
    let exp = Atom::parse("2*(4 + 3*x)*(3 + 2*x + 3*x^2)*(3 + 8*x^2)*(4 + x + x^16)")
        .unwrap()
        .expand();

    let poly: MultivariatePolynomial<_, u8> = exp.to_polynomial(&Z, None);

    let fs = poly.factor();

    println!("Factorization of {}:", poly);
    for (f, _p) in fs {
        println!("\t {}", f);
    }
}

fn factor_univariate_2() {
    let exp = Atom::parse("(x+1)(x+2)(x+3)^3(x+4)(x+5)(x^2+6)(x^3+7)(x+8)^2(x^4+9)(x^5+x+10)")
        .unwrap()
        .expand();

    let poly: MultivariatePolynomial<_, u8> = exp.to_polynomial(&Z, None);

    let fs = poly.factor();

    println!("Factorization of {}:", poly);
    for (f, p) in fs {
        println!("\t {} {}", f, p);
    }
}

fn factor_bivariate() {
    let order = Arc::new(vec![
        Variable::Symbol(Symbol::new("x")),
        Variable::Symbol(Symbol::new("y")),
    ]);

    let input = "(x^2+y+x+1)(3*x+y^2+4)*(6*x*(y+1)+y+5)*(7*x*y+4)";

    let exp = Atom::parse(input).unwrap().expand();

    let poly: MultivariatePolynomial<_, u8> = exp.to_polynomial(&Z, Some(order));

    println!("Factorization of {}:", poly);
    for (f, pow) in poly.factor() {
        println!("\t({})^{}", f, pow);
    }
}

fn factor_multivariate() {
    let order = Arc::new(vec![
        Variable::Symbol(Symbol::new("x")),
        Variable::Symbol(Symbol::new("y")),
        Variable::Symbol(Symbol::new("z")),
        Variable::Symbol(Symbol::new("w")),
    ]);

    let input = "(x*(2+2*y+2*z)+1)*(x*(4+z^2)+y+3)*(x*(w+w^2+4+y)+w+5)";

    let exp = Atom::parse(input).unwrap().expand();

    let poly: MultivariatePolynomial<_, u8> = exp.to_polynomial(&Z, Some(order));

    println!("Factorization of {}:", poly);
    for (f, p) in poly.factor() {
        println!("\t({})^{}", f, p);
    }
}

fn main() {
    factor_square_free();
    factor_ff_square_free();
    factor_ff_univariate();
    factor_ff_bivariate();
    factor_univariate_1();
    factor_univariate_2();
    factor_bivariate();
    factor_multivariate();
}
