use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    id::Match,
    parse, symbol,
};

#[test]
fn fibonacci() {
    // prepare all patterns
    let pattern = parse!("f(x_)").unwrap().to_pattern();
    let rhs = parse!("f(x_ - 1) + f(x_ - 2)").unwrap().to_pattern();
    let lhs_zero_pat = parse!("f(0)").unwrap().to_pattern();
    let lhs_one_pat = parse!("f(1)").unwrap().to_pattern();
    let rhs_one = Atom::new_num(1).to_pattern();

    // prepare the pattern restriction `x_ > 1`
    let restrictions = symbol!("x_").filter(|v: &Match| match v {
        Match::Single(AtomView::Num(n)) => !n.is_one() && !n.is_zero(),
        _ => false,
    });

    let mut target = parse!("f(10)").unwrap();

    for _ in 0..9 {
        target = target
            .replace(&pattern)
            .when(&restrictions)
            .with(&rhs)
            .expand()
            .replace(&lhs_zero_pat)
            .with(&rhs_one)
            .replace(&lhs_one_pat)
            .with(&rhs_one);
    }

    assert_eq!(target, Atom::new_num(89));
}

#[test]
fn replace_once() {
    let expr = parse!("f(z)*f(f(x))*f(y)").unwrap();
    let pat_expr = parse!("f(x_)").unwrap();

    let rhs_expr = parse!("g(x_)").unwrap();
    let rhs = rhs_expr.as_view().to_pattern();

    let pattern = pat_expr.as_view().to_pattern();

    let r: Vec<_> = expr.replace(&pattern).iter(&rhs).into_iter().collect();

    let res = [
        "g(z)*f(y)*f(f(x))",
        "f(z)*g(y)*f(f(x))",
        "f(z)*f(y)*g(f(x))",
        "f(z)*f(y)*f(g(x))",
    ];

    let res = res.iter().map(|x| parse!(x).unwrap()).collect::<Vec<_>>();

    assert_eq!(r, res);
}
