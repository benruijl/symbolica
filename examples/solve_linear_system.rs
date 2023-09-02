use symbolica::{
    parser::Token,
    printer::{PrintOptions, RationalPolynomialPrinter},
    rings::{
        integer::IntegerRing,
        linear_system::Matrix,
        rational::RationalField,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::default();

    let system = [["c", "c+1", "c^2+5"], ["1", "c", "c+1"], ["c-1", "-1", "c"]];
    let rhs = ["1", "2", "-1"];

    println!("Solve linear system in x\u{20D7}:");
    for (r, v) in system.iter().zip(&rhs) {
        println!("\t ({}).x\u{20D7} = {}", r.join(","), v);
    }

    let var_map = vec![state.get_or_insert_var("c")];

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
                    RationalField::new(),
                    IntegerRing::new(),
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
                    RationalField::new(),
                    IntegerRing::new(),
                    Some(&var_map),
                )
                .unwrap()
        })
        .collect();

    let m = Matrix {
        shape: (system.len() as u32, system.len() as u32),
        data: system_rat.into(),
        field: RationalPolynomialField::new(IntegerRing::new()),
    };
    let b = Matrix {
        shape: (rhs.len() as u32, 1),
        data: rhs_rat.into(),
        field: RationalPolynomialField::new(IntegerRing::new()),
    };

    match m.solve(&b) {
        Ok(sol) => {
            println!(
                "x\u{20D7} = {{{}}}",
                sol.data
                    .iter()
                    .map(|r| format!(
                        "{}",
                        RationalPolynomialPrinter {
                            poly: r,
                            state: &state,
                            opts: PrintOptions::default(),
                        }
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Err(e) => panic!("Could not solve {:?}", e),
    }
}
