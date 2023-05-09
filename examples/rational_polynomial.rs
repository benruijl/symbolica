use symbolica::{
    parser::parse,
    printer::{PrintMode, RationalPolynomialPrinter},
    representations::{default::DefaultRepresentation, OwnedAtom},
    rings::rational_polynomial::RationalPolynomial,
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("(x*y^2*5+5)^2/(2*x+5)+(x+4)/(6*x^2+1)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let rat: RationalPolynomial<u8> = expr
        .to_view()
        .to_rational_polynomial(&workspace, &state, None)
        .unwrap();
    println!(
        "{}",
        RationalPolynomialPrinter::new(&rat, &state, PrintMode::default())
    );
}
