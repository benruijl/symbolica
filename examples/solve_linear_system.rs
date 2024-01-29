use std::sync::Arc;

use symbolica::{
    domains::{
        integer::IntegerRing,
        linear_system::Matrix,
        rational::RationalField,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    parser::Token,
    poly::Variable,
    representations::{Atom, AtomView},
    state::{State, Workspace},
};

fn solve() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::default();

    let x = state.get_or_insert_var("x");
    let y = state.get_or_insert_var("y");
    let z = state.get_or_insert_var("z");
    let eqs = ["c*x + f(c)*y + z - 1", "x + c*y + z/c - 2", "(c-1)x + c*z"];

    let atoms: Vec<_> = eqs
        .iter()
        .map(|e| Atom::parse(e, &mut state, &workspace).unwrap())
        .collect();
    let system: Vec<_> = atoms.iter().map(|x| x.as_view()).collect();

    let sol = AtomView::solve_linear_system::<u8>(&system, &[x, y, z], &workspace, &state).unwrap();

    for (v, s) in ["x", "y", "z"].iter().zip(&sol) {
        println!("{} = {}", v, s.printer(&state));
    }
}

fn solve_from_matrix() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::default();

    let system = [["c", "c+1", "c^2+5"], ["1", "c", "c+1"], ["c-1", "-1", "c"]];
    let rhs = ["1", "2", "-1"];

    println!("Solve linear system in x\u{20D7}:");
    for (r, v) in system.iter().zip(&rhs) {
        println!("\t ({}).x\u{20D7} = {}", r.join(","), v);
    }

    let var_map = Arc::new(vec![Variable::Identifier(state.get_or_insert_var("c"))]);

    let system_rat: Vec<RationalPolynomial<IntegerRing, u8>> = system
        .iter()
        .flatten()
        .map(|s| {
            Token::parse(s)
                .unwrap()
                .to_atom(&mut state, &workspace)
                .unwrap()
                .as_view()
                .to_rational_polynomial(
                    &workspace,
                    &state,
                    &RationalField::new(),
                    &IntegerRing::new(),
                    Some(&var_map),
                )
                .unwrap()
        })
        .collect();

    let rhs_rat: Vec<RationalPolynomial<IntegerRing, u8>> = rhs
        .iter()
        .map(|s| {
            Token::parse(s)
                .unwrap()
                .to_atom(&mut state, &workspace)
                .unwrap()
                .as_view()
                .to_rational_polynomial(
                    &workspace,
                    &state,
                    &RationalField::new(),
                    &IntegerRing::new(),
                    Some(&var_map),
                )
                .unwrap()
        })
        .collect();

    let m = Matrix {
        shape: (system.len() as u32, system.len() as u32),
        data: system_rat.into(),
        field: RationalPolynomialField::new_from_poly(&rhs_rat[0].numerator),
    };
    let b = Matrix {
        shape: (rhs.len() as u32, 1),
        field: RationalPolynomialField::new_from_poly(&rhs_rat[0].numerator),
        data: rhs_rat.into(),
    };

    match m.solve(&b) {
        Ok(sol) => {
            println!(
                "x\u{20D7} = {{{}}}",
                sol.data
                    .iter()
                    .map(|r| format!("{}", r.printer(&state)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Err(e) => panic!("Could not solve {:?}", e),
    }
}

fn main() {
    solve();
    solve_from_matrix();
}
