use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    id::{Match, Pattern, WildcardRestriction},
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
        Symbol::new("x_"),
        WildcardRestriction::Filter(Box::new(|v: &Match| match v {
            Match::Single(AtomView::Num(n)) => !n.is_one() && !n.is_zero(),
            _ => false,
        })),
    )
        .into();

    let mut target = Atom::parse("f(10)").unwrap();

    println!(
        "> Repeated calls of f(x_) = f(x_ - 1) + f(x_ - 2) on {}:",
        target,
    );

    for _ in 0..9 {
        let out = target
            .replace_all(&pattern, &rhs, Some(&restrictions), None)
            .expand()
            .replace_all(&lhs_zero_pat, &rhs_one, None, None)
            .replace_all(&lhs_one_pat, &rhs_one, None, None);

        println!("\t{}", out);

        target = out;
    }
}
