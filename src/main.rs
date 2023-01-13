use symbolica::{
    printer::AtomPrinter,
    representations::{
        default::OwnedAtom,
        tree::{Atom, Number},
        OwnedAtomT,
    },
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();

    // create variable names
    let (x, y, z) = (
        state.get_or_insert("x"),
        state.get_or_insert("y"),
        state.get_or_insert("z"),
    );

    // create term
    let a = Atom::Term(vec![
        Atom::Var(x, Number::new(2, 1)),
        Atom::Number(Number::new(2000, 1)),
        Atom::Var(y, Number::new(2, 1)),
        Atom::Var(x, Number::new(1, 2)),
        Atom::Number(Number::new(4, 4000)),
        Atom::Fn(z, vec![Atom::Var(x, Number::new(3, 4))]),
    ]);

    let b = OwnedAtom::from_tree(&a);

    assert!(
        a == b.to_tree(),
        "Not similar: {:?} vs {:?}",
        a,
        b.to_tree()
    );

    let view = b.to_view();

    println!(
        "input = {}, atom bytes = {}, rep bytes = {}",
        AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state),
        a.len(),
        b.len()
    );

    let mut workspace = Workspace::new();

    let mut normalized = OwnedAtom::new();

    view.normalize(&mut workspace, &mut normalized);

    println!(
        "out = {}, rep bytes = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        normalized.len()
    );
}
