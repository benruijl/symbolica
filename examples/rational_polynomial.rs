use symbolica::{
    representations::Atom,
    rings::{
        integer::IntegerRing, rational::RationalField, rational_polynomial::RationalPolynomial,
    },
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::default();

    let expr = Atom::parse(
        "(x*y^2*5+5)^2/(2*x+5)+(x+4)/(6*x^2+1)",
        &mut state,
        &workspace,
    )
    .unwrap();
    let rat: RationalPolynomial<IntegerRing, u8> = expr
        .as_view()
        .to_rational_polynomial(
            &workspace,
            &state,
            RationalField::new(),
            IntegerRing::new(),
            None,
        )
        .unwrap();
    println!("{}", rat.printer(&state));
}
