use symbolica::{
    poly::{factor::Factorize, polynomial::MultivariatePolynomial},
    representations::Atom,
    rings::{
        finite_field::{FiniteField, FiniteFieldCore},
        integer::IntegerRing,
    },
    state::{ResettableBuffer, State, Workspace},
};

fn factor_ff() {
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

fn factor() {
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

fn main() {
    factor();
    factor_ff();
}
