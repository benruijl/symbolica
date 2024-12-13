use symbolica::atom::{Atom, AtomCore};

fn main() {
    let expr = Atom::parse("f(z)*f(f(x))*f(y)").unwrap();
    let pat_expr = Atom::parse("f(x_)").unwrap();

    let rhs_expr = Atom::parse("g(x_)").unwrap();
    let rhs = rhs_expr.as_view().to_pattern();

    let pattern = pat_expr.as_view().to_pattern();

    println!(
        "> Replace once {}={} in {}:",
        pat_expr.as_view(),
        rhs_expr.as_view(),
        expr.as_view()
    );

    for x in expr.replace_iter(&pattern, &rhs, None, None) {
        println!("\t{}", x);
    }
}
