use symbolica::{
    atom::{Atom, AtomCore},
    domains::rational::Rational,
    evaluate::{CompileOptions, FunctionMap, InlineASM, OptimizationSettings},
    parse, symb,
};

fn main() {
    let e1 = parse!("x + pi + cos(x) + f(g(x+1),h(x*2)) + p(1,x)").unwrap();
    let e2 = parse!("x + h(x*2) + cos(x)").unwrap();
    let f = parse!("y^2 + z^2*y^2").unwrap();
    let g = parse!("i(y+7)+x*i(y+7)*(y-1)").unwrap();
    let h = parse!("y*(1+x*(1+x^2)) + y^2*(1+x*(1+x^2))^2 + 3*(1+x^2)").unwrap();
    let i = parse!("y - 1").unwrap();
    let p1 = parse!("3*z^3 + 4*z^2 + 6*z +8").unwrap();

    let mut fn_map = FunctionMap::new();

    fn_map.add_constant(Atom::new_var(symb!("pi")), Rational::from((22, 7)).into());
    fn_map
        .add_tagged_function(
            symb!("p"),
            vec![Atom::new_num(1).into()],
            "p1".to_string(),
            vec![symb!("z")],
            p1,
        )
        .unwrap();
    fn_map
        .add_function(symb!("f"), "f".to_string(), vec![symb!("y"), symb!("z")], f)
        .unwrap();
    fn_map
        .add_function(symb!("g"), "g".to_string(), vec![symb!("y")], g)
        .unwrap();
    fn_map
        .add_function(symb!("h"), "h".to_string(), vec![symb!("y")], h)
        .unwrap();
    fn_map
        .add_function(symb!("i"), "i".to_string(), vec![symb!("y")], i)
        .unwrap();

    let params = vec![parse!("x").unwrap()];

    let evaluator = Atom::evaluator_multiple(
        &[e1.as_view(), e2.as_view()],
        &fn_map,
        &params,
        OptimizationSettings::default(),
    )
    .unwrap();

    let mut e_f64 = evaluator.map_coeff(&|x| x.into());
    let r = e_f64.evaluate_single(&[5.]);
    println!("{}", r);

    let mut compiled = e_f64
        .export_cpp("nested_evaluate.cpp", "nested", true, InlineASM::X64)
        .unwrap()
        .compile("nested", CompileOptions::default())
        .unwrap()
        .load()
        .unwrap();

    let mut out = vec![0.];
    compiled.evaluate(&[5.], &mut out);
    println!("{}", out[0]);
}
