use symbolica::{
    poly::{factor::Factorize, polynomial::MultivariatePolynomial},
    representations::Atom,
    rings::{
        finite_field::{FiniteField, FiniteFieldCore},
        integer::IntegerRing,
    },
    state::{ResettableBuffer, State, Workspace},
};

fn factor_ff_univariate() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();
    let mut exp = Atom::new();
    Atom::parse("x^100-1", &mut state, &workspace)
        .unwrap()
        .as_view()
        .expand(&workspace, &mut state, &mut exp);

    let field = FiniteField::<u32>::new(17);
    let poly: MultivariatePolynomial<_, u8> = exp.as_view().to_polynomial(field, None).unwrap();

    let factors = poly.square_free_factorization();

    println!("Factorization of {}:", poly.printer(&state));
    for (f, pow) in factors {
        println!("\t({})^{}", f.printer(&state), pow);
    }
}

fn factor_ff_square_free() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();
    let mut exp = Atom::new();
    Atom::parse("(1+x)*(1+x^2)^2*(x^4+1)^3", &mut state, &workspace)
        .unwrap()
        .as_view()
        .expand(&workspace, &mut state, &mut exp);

    let field = FiniteField::<u32>::new(3);
    let poly: MultivariatePolynomial<_, u8> = exp.as_view().to_polynomial(field, None).unwrap();

    let factors = poly.square_free_factorization();

    println!("Square-free factorization of {}:", poly.printer(&state));
    for (f, pow) in factors {
        println!("\t({})^{}", f.printer(&state), pow);
    }
}

fn factor_square_free() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();
    let mut exp = Atom::new();
    Atom::parse("3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)", &mut state, &workspace)
        .unwrap()
        .as_view()
        .expand(&workspace, &mut state, &mut exp);

    let poly: MultivariatePolynomial<_, u8> = exp
        .as_view()
        .to_polynomial(IntegerRing::new(), None)
        .unwrap();

    let factors = poly.square_free_factorization();

    println!("Square-free factorization of {}:", poly.printer(&state));
    for (f, pow) in factors {
        println!("\t({})^{}", f.printer(&state), pow);
    }
}

fn factor_univariate_1() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();
    let mut exp = Atom::new();
    Atom::parse(
        "2*(4 + 3*x)*(3 + 2*x + 3*x^2)*(3 + 8*x^2)*(4 + x + x^16)",
        &mut state,
        &workspace,
    )
    .unwrap()
    .as_view()
    .expand(&workspace, &mut state, &mut exp);

    let poly: MultivariatePolynomial<_, u8> = exp
        .as_view()
        .to_polynomial(IntegerRing::new(), None)
        .unwrap();

    let fs = poly.factor_univariate();

    println!("Factorization of {}:", poly.printer(&state));
    for (f, _p) in fs {
        println!("\t {}", f.printer(&state));
    }
}

fn factor_univariate_2() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();
    let mut exp = Atom::new();
    Atom::parse(
        "(x+1)(x+2)(x+3)^3(x+4)(x+5)(x^2+6)(x^3+7)(x+8)^2(x^4+9)(x^5+x+10)",
        &mut state,
        &workspace,
    )
    .unwrap()
    .as_view()
    .expand(&workspace, &mut state, &mut exp);

    let poly: MultivariatePolynomial<_, u8> = exp
        .as_view()
        .to_polynomial(IntegerRing::new(), None)
        .unwrap();

    let fs = poly.factor_univariate();

    println!("Factorization of {}:", poly.printer(&state));
    for (f, p) in fs {
        println!("\t {} {}", f.printer(&state), p);
    }
}

fn main() {
    factor_square_free();
    factor_ff_square_free();
    factor_ff_univariate();
    factor_univariate_1();
    factor_univariate_2();
}
