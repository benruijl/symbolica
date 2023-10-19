use symbolica::{
    poly::{
        groebner::{GrevLexOrder, GroebnerBasis},
        polynomial::MultivariatePolynomial,
    },
    representations::Atom,
    rings::finite_field::{FiniteField, FiniteFieldCore},
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();

    for x in 'a'..'z' {
        state.get_or_insert_var(x.to_string());
    }

    // cyclic-4
    let polys = [
        "a b c d - 1",
        "a b c + a b d + a c d + b c d",
        "a b + b c + a d + c d",
        "a + b + c + d",
    ];

    let fs: Vec<MultivariatePolynomial<_, u16>> = polys
        .iter()
        .map(|x| {
            let a = Atom::parse(x, &mut state, &workspace).unwrap();
            let mut res = workspace.new_atom();
            a.as_view().expand(&workspace, &state, &mut res);
            res.as_view()
                .to_polynomial(FiniteField::<u32>::new(13), None)
                .unwrap()
        })
        .collect();

    // compute the Groebner basis with grevlex ordering
    let gb = GroebnerBasis::<_, _, GrevLexOrder>::new(&fs, true);

    println!("Basis:");
    for g in &gb.system {
        let p: MultivariatePolynomial<_, _> = g.into();
        println!("\t{}", p.printer(&state));
    }
}
