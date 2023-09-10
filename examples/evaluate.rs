use ahash::HashMap;
use symbolica::evaluate::EvaluationFn;

use symbolica::poly::evaluate::InstructionSetPrinter;
use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::printer::{PolynomialPrinter, PrintOptions};
use symbolica::rings::rational::RationalField;
use symbolica::{
    representations::{Atom, Identifier},
    state::{State, Workspace},
};

/// Convert an expression to a polynomial with a map for all the non-polynomial
/// parts. Then use the generic evaluator to evaluate all non-polynomial parts
/// and create the input to evaluate the optimized polynomial.
fn from_poly() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();

    let a = Atom::parse(
        "x^2+x*f(x) + 1 / x + cos(x)^2 + f(x)*x^3",
        &mut state,
        &workspace,
    )
    .unwrap();

    let mut non_poly_map = HashMap::default();
    let p: MultivariatePolynomial<_, u8> = a.as_view().to_polynomial_with_map(
        "N",
        &mut state,
        RationalField::new(),
        None,
        &mut non_poly_map,
    );

    // invert map
    let id_to_non_poly: HashMap<_, _> = non_poly_map.into_iter().map(|(k, v)| (v, k)).collect();

    println!(
        "poly = {}",
        PolynomialPrinter::new(&p, &state, PrintOptions::default())
    );
    for (i, a) in &id_to_non_poly {
        println!("{} = {}", state.get_name(*i).unwrap(), a.printer(&state));
    }

    let (h, _ops, _scheme) = p.optimize_horner_scheme(4000);
    let mut i = h.to_instr(p.nvars);
    i.fuse_operations();
    while i.common_pair_elimination() {
        i.fuse_operations();
    }

    let out = i.to_output(true);
    println!(
        "Optimized output:\n{}",
        InstructionSetPrinter {
            instr: &out,
            var_map: p.var_map.as_ref().unwrap(),
            state: &state,
            mode: symbolica::poly::evaluate::InstructionSetMode::Plain,
        }
    );
    let mut evaluator = out.convert::<f64>().evaluator();

    let mut var_map = HashMap::default();
    let mut fn_map: HashMap<Identifier, EvaluationFn<_, _>> = HashMap::default();
    let mut cache = HashMap::default();

    // x = 6.
    let x = state.get_or_insert_var("x");
    var_map.insert(x, 6.);

    // f(x) = x^2
    let f = state.get_or_insert_var("f");
    fn_map.insert(
        f,
        EvaluationFn::new(Box::new(|args: &[f64], _, _, _| args[0] * args[0])),
    );

    // determine the input of the polynomial by calling the evaluation function
    let input: Vec<_> = p
        .var_map
        .unwrap()
        .iter()
        .map(|var| {
            var_map.get(var).cloned().unwrap_or_else(|| {
                id_to_non_poly
                    .get(var)
                    .unwrap()
                    .evaluate::<f64>(&var_map, &fn_map, &mut cache)
            })
        })
        .collect();

    println!("Evaluation = {}", evaluator.evaluate(&input)[0]);
}

/// Evaluate a generic expression.
fn from_atom() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();

    let x = state.get_or_insert_var("x");
    let f = state.get_or_insert_var("f");
    let g = state.get_or_insert_var("g");
    let a = Atom::parse("x*cos(x) + f(x, 1)^2 + g(g(x))", &mut state, &workspace).unwrap();

    let mut var_map = HashMap::default();
    let mut fn_map: HashMap<Identifier, EvaluationFn<_, _>> = HashMap::default();
    let mut cache = HashMap::default();

    // x = 6.
    var_map.insert(x, 6.);

    // f(x, y) = x^2 + y
    fn_map.insert(
        f,
        EvaluationFn::new(Box::new(|args: &[f64], _, _, _| {
            args[0] * args[0] + args[1]
        })),
    );

    // g(x) = f(x, 3)
    fn_map.insert(
        g,
        EvaluationFn::new(Box::new(move |args: &[f64], var_map, fn_map, cache| {
            fn_map.get(&f).unwrap().get()(&[args[0], 3.], var_map, fn_map, cache)
        })),
    );

    println!(
        "Result for x = 6.: {}",
        a.as_view().evaluate::<f64>(&var_map, &fn_map, &mut cache)
    );
}

fn main() {
    from_atom();
    from_poly();
}
