use symbolica::{
    id::AtomTreeIterator,
    representations::Atom,
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: Atom = Atom::parse("f(z)*f(f(x),z)*f(y)", &mut state, &workspace).unwrap();

    println!("> Tree walk of {}:", expr.printer(&state));

    for (loc, view) in AtomTreeIterator::new(expr.as_view()) {
        println!("\tAtom at location {:?}: {}", loc, view.printer(&state));
    }
}
