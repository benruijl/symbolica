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
        Atom::Pow(Box::new((Atom::Var(x), Atom::Number(Number::new(2, 1))))),
        Atom::Number(Number::new(3, 1)),
        Atom::Pow(Box::new((Atom::Var(x), Atom::Number(Number::new(1, 2))))),
        Atom::Var(y),
        Atom::Number(Number::new(4, 4000)),
        Atom::Var(y),
        Atom::Fn(
            z,
            vec![Atom::Expression(vec![
                Atom::Number(Number::new(1, 1)),
                Atom::Var(x),
            ])],
        ),
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
