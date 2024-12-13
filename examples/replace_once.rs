use symbolica::{
    atom::{Atom, AtomCore},
    id::{Condition, MatchSettings},
};

fn main() {
    let expr = Atom::parse("f(z)*f(f(x))*f(y)").unwrap();
    let pat_expr = Atom::parse("f(x_)").unwrap();

    let rhs_expr = Atom::parse("g(x_)").unwrap();
    let rhs = rhs_expr.as_view().to_pattern().into();

    let pattern = pat_expr.as_view().to_pattern();
    let restrictions = Condition::default();
    let settings = MatchSettings::default();

    println!(
        "> Replace once {}={} in {}:",
        pat_expr.as_view(),
        rhs_expr.as_view(),
        expr.as_view()
    );

    let mut replaced = Atom::new();

    let mut it = expr.replace_iter(&pattern, &rhs, &restrictions, &settings);
    while let Some(()) = it.next(&mut replaced) {
        println!("\t{}", replaced);
    }
}
