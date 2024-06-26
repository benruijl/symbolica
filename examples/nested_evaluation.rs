use ahash::HashMap;
use symbolica::{atom::Atom, evaluate::ConstOrExpr, state::State};

fn main() {
    let e = Atom::parse("x + cos(x) + f(g(x+1),h(x*2)) + p(1)").unwrap();
    let f = Atom::parse("y^2 + z^2").unwrap(); // f(y,z) = y^2+z^2
    let g = Atom::parse("i(y+7)").unwrap(); // g(y) = i(y+7)
    let h = Atom::parse("y + 3").unwrap(); // h(y) = y+3
    let i = Atom::parse("y * 2").unwrap(); // i(y) = y*2
    let k = Atom::parse("x+8").unwrap(); // p(1) = x + 8

    let mut const_map = HashMap::default();

    let p1 = Atom::parse("p(1)").unwrap();
    let f_s = Atom::new_var(State::get_symbol("f"));
    let g_s = Atom::new_var(State::get_symbol("g"));
    let h_s = Atom::new_var(State::get_symbol("h"));
    let i_s = Atom::new_var(State::get_symbol("i"));

    const_map.insert(p1.into(), ConstOrExpr::Expr(vec![], k.as_view()));

    const_map.insert(
        f_s.into(),
        ConstOrExpr::Expr(
            vec![State::get_symbol("y"), State::get_symbol("z")],
            f.as_view(),
        ),
    );
    const_map.insert(
        g_s.into(),
        ConstOrExpr::Expr(vec![State::get_symbol("y")], g.as_view()),
    );
    const_map.insert(
        h_s.into(),
        ConstOrExpr::Expr(vec![State::get_symbol("y")], h.as_view()),
    );
    const_map.insert(
        i_s.into(),
        ConstOrExpr::Expr(vec![State::get_symbol("y")], i.as_view()),
    );

    let params = vec![Atom::parse("x").unwrap()];

    let mut evaluator = e.as_view().evaluator(|r| r.into(), &const_map, &params);

    println!("{}", evaluator.evaluate(&[5.]));
}
