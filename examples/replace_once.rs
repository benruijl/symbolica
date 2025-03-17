use symbolica::{atom::AtomCore, parse};

fn main() {
    let expr = parse!("f(z)*f(f(x))*f(y)").unwrap();
    let pat_expr = parse!("f(x_)").unwrap();

    let rhs_expr = parse!("g(x_)").unwrap();
    let rhs = rhs_expr.to_pattern();

    let pattern = pat_expr.to_pattern();

    println!("> Replace once {}={} in {}:", pat_expr, rhs_expr, expr);

    for x in expr.replace(pattern).iter(rhs) {
        println!("\t{}", x);
    }
}
