use std::time::Instant;

use symbolica::{
    atom::{Atom, AtomView},
    domains::{float::Complex, rational::Rational},
    evaluate::{CompileOptions, ExpressionEvaluator, FunctionMap},
    state::State,
};

fn main() {
    let e1 = Atom::parse("x + pi + cos(x) + f(g(x+1),h(x*2)) + p(1,x)").unwrap();
    let e2 = Atom::parse("x + h(x*2) + cos(x)").unwrap();
    let f = Atom::parse("y^2 + z^2*y^2").unwrap();
    let g = Atom::parse("i(y+7)+x*i(y+7)*(y-1)").unwrap();
    let h = Atom::parse("y*(1+x*(1+x^2)) + y^2*(1+x*(1+x^2))^2 + 3*(1+x^2)").unwrap();
    let i = Atom::parse("y - 1").unwrap();
    let p1 = Atom::parse("3*z^3 + 4*z^2 + 6*z +8").unwrap();

    let mut fn_map = FunctionMap::new();

    fn_map.add_constant(
        Atom::new_var(State::get_symbol("pi")).into(),
        Rational::from((22, 7)).into(),
    );
    fn_map
        .add_tagged_function(
            State::get_symbol("p"),
            vec![Atom::new_num(1).into()],
            "p1".to_string(),
            vec![State::get_symbol("z")],
            p1.as_view(),
        )
        .unwrap();
    fn_map
        .add_function(
            State::get_symbol("f"),
            "f".to_string(),
            vec![State::get_symbol("y"), State::get_symbol("z")],
            f.as_view(),
        )
        .unwrap();
    fn_map
        .add_function(
            State::get_symbol("g"),
            "g".to_string(),
            vec![State::get_symbol("y")],
            g.as_view(),
        )
        .unwrap();
    fn_map
        .add_function(
            State::get_symbol("h"),
            "h".to_string(),
            vec![State::get_symbol("y")],
            h.as_view(),
        )
        .unwrap();
    fn_map
        .add_function(
            State::get_symbol("i"),
            "i".to_string(),
            vec![State::get_symbol("y")],
            i.as_view(),
        )
        .unwrap();

    let params = vec![Atom::parse("x").unwrap()];

    let mut tree = AtomView::to_eval_tree_multiple(
        &[e1.as_view(), e2.as_view()],
        |r| r.clone(),
        &fn_map,
        &params,
    )
    .unwrap();

    // optimize the tree using an occurrence-order Horner scheme
    println!("Op original {:?}", tree.count_operations());
    tree.horner_scheme();
    println!("Op horner {:?}", tree.count_operations());
    tree.common_subexpression_elimination();
    println!("op cse {:?}", tree.count_operations());

    tree.common_pair_elimination();
    println!("op cpe {:?}", tree.count_operations());

    let ce = tree
        .export_cpp("nested_evaluation.cpp", "evaltest", true)
        .unwrap()
        .compile("libneval.so", CompileOptions::default())
        .unwrap()
        .load()
        .unwrap();

    let params = vec![5.];
    let mut out = vec![0., 0.];
    ce.evaluate(&params, &mut out);
    println!("Eval from C++: {}, {}", out[0], out[1]);

    {
        let params = vec![Complex::new(5., 0.)];
        let mut out = vec![Complex::new_zero(), Complex::new_zero()];
        ce.evaluate(&params, &mut out);
        println!("Eval from C++: {}, {}", out[0], out[1]);
    }

    // benchmark
    let t = Instant::now();
    for _ in 0..1000000 {
        let _ = ce.evaluate(&params, &mut out);
    }
    println!("C++ time {:#?}", t.elapsed());

    let t2 = tree.map_coeff::<f64, _>(&|r| r.into());
    let mut evaluator: ExpressionEvaluator<f64> = t2.linearize(params.len());

    evaluator.evaluate_multiple(&params, &mut out);
    println!("Eval: {}, {}", out[0], out[1]);

    let params = vec![5.];
    let t = Instant::now();
    for _ in 0..1000000 {
        evaluator.evaluate_multiple(&params, &mut out);
    }
    println!("Eager time {:#?}", t.elapsed());
}
