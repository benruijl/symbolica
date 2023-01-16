use symbolica::{
    printer::AtomPrinter,
    representations::{
        default::OwnedAtomD,
        tree::{AtomTree, Number},
        OwnedAtom,
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
    let a = AtomTree::Mul(vec![
        AtomTree::Pow(Box::new((AtomTree::Var(x), AtomTree::Num(Number::new(2, 1))))),
        AtomTree::Num(Number::new(3, 1)),
        AtomTree::Pow(Box::new((AtomTree::Var(x), AtomTree::Num(Number::new(1, 2))))),
        AtomTree::Var(y),
        AtomTree::Num(Number::new(4, 4000)),
        AtomTree::Var(y),
        AtomTree::Fn(
            z,
            vec![AtomTree::Add(vec![
                AtomTree::Num(Number::new(1, 1)),
                AtomTree::Var(x),
            ])],
        ),
    ]);

    let b = OwnedAtomD::from_tree(&a);

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

    let mut normalized = OwnedAtomD::new();

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
