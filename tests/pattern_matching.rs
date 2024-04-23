use symbolica::{
    atom::{Atom, AtomView},
    id::{Condition, Match, MatchSettings, Pattern, PatternRestriction},
    state::{RecycledAtom, State},
};

#[test]
fn fibonacci() {
    // prepare all patterns
    let pattern = Pattern::parse("f(x_)").unwrap();
    let rhs = Pattern::parse("f(x_ - 1) + f(x_ - 2)").unwrap();
    let lhs_zero_pat = Pattern::parse("f(0)").unwrap();
    let lhs_one_pat = Pattern::parse("f(1)").unwrap();
    let rhs_one = Atom::new_num(1).into_pattern();

    // prepare the pattern restriction `x_ > 1`
    let restrictions = (
        State::get_symbol("x_"),
        PatternRestriction::Filter(Box::new(|v: &Match| match v {
            Match::Single(AtomView::Num(n)) => !n.is_one() && !n.is_zero(),
            _ => false,
        })),
    )
        .into();

    let input = Atom::parse("f(10)").unwrap();
    let mut target: RecycledAtom = input.clone().into();

    for _ in 0..9 {
        let mut out = RecycledAtom::new();
        pattern.replace_all_into(target.as_view(), &rhs, Some(&restrictions), None, &mut out);

        let mut out2 = RecycledAtom::new();
        out.expand_into(&mut out2);

        lhs_zero_pat.replace_all_into(out2.as_view(), &rhs_one, None, None, &mut out);

        lhs_one_pat.replace_all_into(out.as_view(), &rhs_one, None, None, &mut out2);

        target = out2;
    }

    assert_eq!(target.into_inner(), Atom::new_num(89));
}

#[test]
fn replace_once() {
    let expr = Atom::parse("f(z)*f(f(x))*f(y)").unwrap();
    let pat_expr = Atom::parse("f(x_)").unwrap();

    let rhs_expr = Atom::parse("g(x_)").unwrap();
    let rhs = rhs_expr.as_view().into_pattern();

    let pattern = pat_expr.as_view().into_pattern();
    let restrictions = Condition::default();
    let settings = MatchSettings::default();

    let mut replaced = Atom::new();

    let mut it = pattern.replace_iter(expr.as_view(), &rhs, &restrictions, &settings);
    let mut r = vec![];
    while let Some(()) = it.next(&mut replaced) {
        r.push(replaced.clone());
    }

    let res = [
        "g(z)*f(y)*f(f(x))",
        "f(z)*g(y)*f(f(x))",
        "f(z)*f(y)*g(f(x))",
        "f(z)*f(y)*f(g(x))",
    ];

    let res = res
        .iter()
        .map(|x| Atom::parse(x).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(r, res);
}
