use ahash::HashMap;

use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::printer::{PolynomialPrinter, PrintOptions};
use symbolica::rings::integer::IntegerRing;
use symbolica::{
    representations::Atom,
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();
    let a = Atom::parse("x^2+x + 1 / x + f(x)", &mut state, &workspace).unwrap();

    let mut map = HashMap::default();
    let p: MultivariatePolynomial<_, u8> =
        a.as_view()
            .to_polynomial_with_map("N", &mut state, IntegerRing::new(), None, &mut map);

    println!(
        "poly = {}",
        PolynomialPrinter::new(&p, &state, PrintOptions::default())
    );
    for (a, i) in map {
        println!("{} = {}", state.get_name(i).unwrap(), a.printer(&state));
    }
}
