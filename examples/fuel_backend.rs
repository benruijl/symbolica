use std::io::{self, BufRead, Write};

use smartstring::{LazyCompact, SmartString};
use symbolica::{
    parser::parse,
    printer::{PrintMode, RationalPolynomialPrinter},
    representations::default::DefaultRepresentation,
    rings::{
        integer::IntegerRing, rational::RationalField, rational_polynomial::RationalPolynomial,
    },
    state::{State, Workspace},
};

fn main() {
    let mut buffer = String::with_capacity(2048);
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();

    // read the number of variables
    let _ = stdin.read_line(&mut buffer).unwrap();
    let mut num_vars_and_var_names = buffer.split(' ');
    let num_vars = num_vars_and_var_names
        .next()
        .expect("Expected number of variables")
        .parse::<usize>()
        .expect("Number of vars should be a non-negative integer");

    let mut var_names: Vec<SmartString<LazyCompact>> = vec![];
    for _ in 0..num_vars {
        var_names.push(
            num_vars_and_var_names
                .next()
                .expect("Expected variable")
                .trim_end()
                .into(),
        );
    }

    let mut state = State::new();
    let workspace = Workspace::<DefaultRepresentation>::new();
    let vars: Vec<_> = var_names
        .iter()
        .map(|v| state.get_or_insert_var(v))
        .collect();

    buffer.clear();
    while let Ok(n) = stdin.read_line(&mut buffer) {
        if n == 0 || buffer.starts_with('\n') || buffer.starts_with("&q") {
            break;
        }

        let r: RationalPolynomial<IntegerRing, u16> = parse(&buffer)
            .unwrap()
            .to_rational_polynomial(
                &workspace,
                &mut state,
                RationalField::new(),
                IntegerRing::new(),
                &vars,
                &var_names,
            )
            .unwrap();

        let out_str = format!(
            "{}",
            RationalPolynomialPrinter {
                poly: &r,
                state: &state,
                print_mode: PrintMode::default()
            }
        );

        writeln!(&mut stdout, "{}", out_str).unwrap();

        buffer.clear();
    }
}
