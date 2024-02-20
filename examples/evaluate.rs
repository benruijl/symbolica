use ahash::HashMap;
use symbolica::evaluate::EvaluationFn;
use symbolica::poly::Variable;
use symbolica::{
    representations::Atom,
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();

    let x = state.get_or_insert_var("x").into();
    let f: Variable = state.get_or_insert_var("f").into();
    let g = state.get_or_insert_var("g").into();
    let a = Atom::parse("x*cos(x) + f(x, 1)^2 + g(g(x))", &mut state, &workspace).unwrap();

    let mut var_map = HashMap::default();
    let mut fn_map: HashMap<Variable, EvaluationFn<_, _>> = HashMap::default();
    let mut cache = HashMap::default();

    // x = 6.
    var_map.insert(x, 6.);

    // f(x, y) = x^2 + y
    fn_map.insert(
        f.clone(),
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
