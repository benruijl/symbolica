use symbolica::{
    poly::polynomial::MultivariatePolynomial,
    domains::{
        integer::{Integer, IntegerRing},
        rational::{Rational, RationalField},
    },
};
use tracing_subscriber::{fmt, prelude::*, util::SubscriberInitExt, EnvFilter};

fn gcd_integer_poly() {
    let field = IntegerRing::new();
    let mut a = MultivariatePolynomial::<IntegerRing, u8>::new(2, &field, Some(5), None);
    a.append_monomial(Integer::Natural(5), &[0, 0]);
    a.append_monomial(Integer::Natural(8), &[1, 0]);
    a.append_monomial(Integer::Natural(3), &[2, 0]);
    a.append_monomial(Integer::Natural(-5), &[0, 1]);
    a.append_monomial(Integer::Natural(-3), &[1, 1]);

    let mut b = MultivariatePolynomial::<IntegerRing, u8>::new(2, &field, Some(5), None);
    b.append_monomial(Integer::Natural(5), &[0, 0]);
    b.append_monomial(Integer::Natural(5), &[1, 0]);
    b.append_monomial(Integer::Natural(-2), &[0, 1]);
    b.append_monomial(Integer::Natural(3), &[1, 1]);
    b.append_monomial(Integer::Natural(-3), &[0, 2]);

    println!("> Polynomial gcd of {} and {} =", a, b);
    println!("\t{}", MultivariatePolynomial::gcd(&a, &b));
}

fn gcd_rat_poly() {
    let field = RationalField::new();
    let mut a = MultivariatePolynomial::<RationalField, u8>::new(3, &field, Some(6), None);
    a.append_monomial(Rational::Natural(3, 4), &[2, 0, 0]);
    a.append_monomial(Rational::Natural(3, 4), &[3, 0, 0]);
    a.append_monomial(Rational::Natural(1, 1), &[0, 1, 0]);
    a.append_monomial(Rational::Natural(1, 1), &[1, 1, 0]);
    a.append_monomial(Rational::Natural(1, 1), &[0, 0, 1]);
    a.append_monomial(Rational::Natural(1, 1), &[1, 0, 1]);

    let mut b = MultivariatePolynomial::<RationalField, u8>::new(3, &field, Some(8), None);
    b.append_monomial(Rational::Natural(3, 2), &[2, 0, 0]);
    b.append_monomial(Rational::Natural(2, 1), &[0, 1, 0]);
    b.append_monomial(Rational::Natural(9, 20), &[2, 1, 0]);
    b.append_monomial(Rational::Natural(3, 5), &[0, 2, 0]);
    b.append_monomial(Rational::Natural(2, 1), &[0, 0, 1]);
    b.append_monomial(Rational::Natural(-3, 4), &[2, 0, 1]);
    b.append_monomial(Rational::Natural(-2, 5), &[0, 1, 1]);
    b.append_monomial(Rational::Natural(-1, 1), &[0, 0, 2]);

    println!("> Polynomial gcd of {} and {} =", a, b);
    println!("\t{}", MultivariatePolynomial::gcd(&a, &b));
}

fn main() {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_env("SYMBOLICA_LOG"))
        .init();

    gcd_integer_poly();
    gcd_rat_poly();
}
