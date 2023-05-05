use symbolica::{
    id::AtomTreeIterator,
    parser::parse,
    printer::AtomPrinter,
    representations::{default::DefaultRepresentation, OwnedAtom},
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(z)*f(f(x),z)*f(y)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    println!(
        "> Tree walk of {}:",
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::default(), &state)
    );

    for (loc, view) in AtomTreeIterator::new(expr.to_view()) {
        println!(
            "\tAtom at location {:?}: {}",
            loc,
            AtomPrinter::new(view, symbolica::printer::PrintMode::default(), &state)
        );
    }
}
