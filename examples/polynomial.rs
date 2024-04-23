use std::sync::Arc;

use symbolica::{
    domains::{
        finite_field::{FiniteFieldCore, Zp},
        rational::{Rational, Q},
    },
    poly::polynomial::MultivariatePolynomial,
    state::State,
};

fn main() {
    let x = State::get_symbol("x");
    let y = State::get_symbol("y");
    let z = State::get_symbol("z");
    let vars = Arc::new(vec![x.into(), y.into(), z.into()]);
    let mut a = MultivariatePolynomial::<_, u8>::new(&Q, Some(3), vars.clone());
    a.append_monomial(Rational::Natural(3, 4), &[1, 0, 0]);
    a.append_monomial(Rational::Natural(5, 1), &[1, 1, 0]);
    a.append_monomial(Rational::Natural(7, 3), &[1, 1, 2]);

    let mut b = MultivariatePolynomial::<_, u8>::new(&Q, Some(3), vars.clone());
    b.append_monomial(Rational::Natural(6, 7), &[0, 1, 0]);
    b.append_monomial(Rational::Natural(5, 1), &[1, 1, 0]);
    b.append_monomial(Rational::Natural(7, 3), &[1, 1, 2]);

    println!("> Polynomial multiplication: {} * {} =", a, b);
    println!("\t{}", a * &b);

    let finite_field = Zp::new(17);
    let mut a = MultivariatePolynomial::<_, u8>::new(&finite_field, Some(3), vars.clone());
    a.append_monomial(finite_field.to_element(4), &[1, 0, 0]);
    a.append_monomial(finite_field.to_element(6), &[1, 1, 0]);
    a.append_monomial(finite_field.to_element(13), &[1, 1, 2]);

    let mut b = MultivariatePolynomial::<_, u8>::new(&finite_field, Some(3), vars.clone());
    b.append_monomial(finite_field.to_element(2), &[0, 1, 0]);
    b.append_monomial(finite_field.to_element(1), &[1, 1, 0]);
    b.append_monomial(finite_field.to_element(16), &[1, 1, 2]);

    println!("> Polynomial multiplication: {} * {} =", a, b);
    println!("\t{}", a * &b);
}
