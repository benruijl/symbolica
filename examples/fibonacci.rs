use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    id::{Match, Pattern, WildcardRestriction},
    state::{RecycledAtom, State},
};

fn main() {
    // prepare all patterns
    let pattern = Pattern::parse("f(x_)").unwrap();
    let rhs = Pattern::parse("f(x_ - 1) + f(x_ - 2)").unwrap();
    let lhs_zero_pat = Pattern::parse("f(0)").unwrap();
    let lhs_one_pat = Pattern::parse("f(1)").unwrap();
    let rhs_one = Atom::new_num(1).to_pattern();

    // prepare the pattern restriction `x_ > 1`
    let restrictions = (
        State::get_symbol("x_"),
        WildcardRestriction::Filter(Box::new(|v: &Match| match v {
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
        target.replace_all_into(&pattern, &rhs, Some(&restrictions), None, &mut out);

        let mut out2 = RecycledAtom::new();
        out.expand_into(None, &mut out2);

        out2.replace_all_into(&lhs_zero_pat, &rhs_one, None, None, &mut out);

        out.replace_all_into(&lhs_one_pat, &rhs_one, None, None, &mut out2);

        println!("\t{}", out2);

        target = out2;
    }
}
