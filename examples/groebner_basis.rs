use symbolica::{
    atom::Atom,
    domains::finite_field::Zp,
    poly::{groebner::GroebnerBasis, polynomial::MultivariatePolynomial, GrevLexOrder},
    state::State,
};

fn main() {
    for x in 'a'..='z' {
        State::get_symbol(x.to_string());
    }

    // cyclic-4
    let polys = [
        "a b c d - 1",
        "a b c + a b d + a c d + b c d",
        "a b + b c + a d + c d",
        "a + b + c + d",
    ];

    let ideal: Vec<MultivariatePolynomial<_, u16>> = polys
        .iter()
        .map(|x| {
            let a = Atom::parse(x).unwrap().expand();
            a.to_polynomial(&Zp::new(13), None)
        })
        .collect();

    // compute the Groebner basis with lex ordering
    let gb = GroebnerBasis::new(&ideal, true);

    println!("Lex order basis:");
    for g in &gb.system {
        println!("\t{}", g);
    }

    // compute the Groebner basis with grevlex ordering by converting the polynomials
    let grevlex_ideal: Vec<_> = ideal.iter().map(|p| p.reorder::<GrevLexOrder>()).collect();
    let gb = GroebnerBasis::new(&grevlex_ideal, true);
    println!("Grevlex order basis:");
    for g in &gb.system {
        println!("\t{}", g);
    }
}
