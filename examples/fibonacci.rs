use symbolica::{
    id::{Match, Pattern, PatternRestriction},
    representations::{Atom, AtomView},
    state::{RecycledAtom, State},
};

fn main() {
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

    println!(
        "> Repeated calls of f(x_) = f(x_ - 1) + f(x_ - 2) on {}:",
        target,
    );

    for _ in 0..9 {
        let mut out = RecycledAtom::new();
        pattern.replace_all_into(target.as_view(), &rhs, Some(&restrictions), None, &mut out);

        let mut out2 = RecycledAtom::new();
        out.expand_into(&mut out2);

        lhs_zero_pat.replace_all_into(out2.as_view(), &rhs_one, None, None, &mut out);

        lhs_one_pat.replace_all_into(out.as_view(), &rhs_one, None, None, &mut out2);

        println!("\t{}", out2);

        target = out2;
    }
}
