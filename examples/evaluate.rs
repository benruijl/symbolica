use ahash::HashMap;
use symbolica::evaluate::EvaluationFn;
use symbolica::{representations::Atom, state::State};

fn main() {
    let x = State::get_symbol("x");
    let f = State::get_symbol("f");
    let g = State::get_symbol("g");
    let p0 = Atom::parse("p(0)").unwrap();
    let a = Atom::parse("x*cos(x) + f(x, 1)^2 + g(g(x)) + p(0)").unwrap();

    let mut const_map = HashMap::default();
    let mut fn_map: HashMap<_, EvaluationFn<_>> = HashMap::default();
    let mut cache = HashMap::default();

    // x = 6 and p(0) = 7
    let v = Atom::new_var(x);
    const_map.insert(v.as_view(), 6.);
    const_map.insert(p0.as_view(), 7.);

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
        a.evaluate::<f64>(&const_map, &fn_map, &mut cache)
    );
}
