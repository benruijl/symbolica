use symbolica::{
    atom::{Atom, AtomCore},
    id::{Match, WildcardRestriction},
    parse, symbol,
};

fn main() {
    // prepare all patterns
    let pattern = parse!("f(x_)").unwrap().to_pattern();
    let rhs = parse!("f(x_ - 1) + f(x_ - 2)").unwrap().to_pattern();
    let lhs_zero_pat = parse!("f(0)").unwrap().to_pattern();
    let lhs_one_pat = parse!("f(1)").unwrap().to_pattern();
    let rhs_one = Atom::new_num(1).to_pattern();

    // prepare the pattern restriction `x_ > 1`
    let restrictions =
        symbol!("x_").restrict(WildcardRestriction::filter(|v: &Match| v.to_atom() > 1));

    let mut target = parse!("f(10)").unwrap();

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
