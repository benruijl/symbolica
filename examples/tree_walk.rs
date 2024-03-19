use symbolica::{id::AtomTreeIterator, representations::Atom};

fn main() {
    let expr: Atom = Atom::parse("f(z)*f(f(x),z)*f(y)").unwrap();

    println!("> Tree walk of {}:", expr);

    for (loc, view) in AtomTreeIterator::new(expr.as_view(), (0, None)) {
        println!("\tAtom at location {:?}: {}", loc, view);
    }
}
