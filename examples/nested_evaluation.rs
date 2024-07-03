use std::{process::Command, time::Instant};

use symbolica::{
    atom::{Atom, AtomView},
    domains::rational::Rational,
    evaluate::{ExpressionEvaluator, FunctionMap},
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
    );

    // optimize the tree using an occurrence-order Horner scheme
    println!("Op original {:?}", tree.count_operations());
    tree.horner_scheme();
    println!("Op horner {:?}", tree.count_operations());
    // the compiler seems to do this as well
    tree.common_subexpression_elimination();
    println!("op CSSE {:?}", tree.count_operations());

    tree.common_pair_elimination();
    println!("op CPE {:?}", tree.count_operations());

    let cpp = tree.export_cpp();
    println!("{}", cpp); // print C++ code

    std::fs::write("nested_evaluation.cpp", cpp).unwrap();

    let r = Command::new("g++")
        .arg("-shared")
        .arg("-fPIC")
        .arg("-O3")
        .arg("-ffast-math")
        .arg("-o")
        .arg("libneval.so")
        .arg("nested_evaluation.cpp")
        .output()
        .unwrap();
    println!("Compilation {}", r.status);

    unsafe {
        let lib = libloading::Library::new("./libneval.so").unwrap();
        let func: libloading::Symbol<unsafe extern "C" fn(params: *const f64, out: *mut f64)> =
            lib.get(b"eval_double").unwrap();

        let params = vec![5.];
        let mut out = vec![0., 0.];
        func(params.as_ptr(), out.as_mut_ptr());
        println!("Eval from C++: {}, {}", out[0], out[1]);

        // benchmark

        let t = Instant::now();
        for _ in 0..1000000 {
            let _ = func(params.as_ptr(), out.as_mut_ptr());
        }
        println!("C++ time {:#?}", t.elapsed());
    };

    let t2 = tree.map_coeff::<f64, _>(&|r| r.into());
    let mut evaluator: ExpressionEvaluator<f64> = t2.linearize(params.len());

    let mut out = vec![0., 0.];
    evaluator.evaluate_multiple(&[5.], &mut out);
    println!("Eval: {}, {}", out[0], out[1]);

    // benchmark
    let params = vec![5.];
    let t = Instant::now();
    for _ in 0..1000000 {
        evaluator.evaluate_multiple(&params, &mut out);
    }
    println!("Eager time {:#?}", t.elapsed());
}
