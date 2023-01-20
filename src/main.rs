use symbolica::{
    printer::AtomPrinter,
    representations::{
        tree::{AtomTree, Number},
        Mul, OwnedAtom, OwnedMul,
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
        AtomTree::Pow(Box::new((
            AtomTree::Var(x),
            AtomTree::Num(Number::new(2, 1)),
        ))),
        AtomTree::Num(Number::new(3, 1)),
        AtomTree::Pow(Box::new((
            AtomTree::Var(x),
            AtomTree::Num(Number::new(-2, 1)),
        ))),
        AtomTree::Var(y),
        AtomTree::Num(Number::new(-5, 6000)),
        AtomTree::Var(y),
        AtomTree::Fn(
            z,
            vec![AtomTree::Add(vec![
                AtomTree::Num(Number::new(1, 1)),
                AtomTree::Var(x),
            ])],
        ),
    ]);

    let mut b = OwnedAtom::new();
    b.from_tree(&a);

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

    let workspace = Workspace::new();

    let mut normalized_handle = workspace.get_atom_stack();

    let normalized = normalized_handle.get_buf_mut().transform_to_mul();

    view.normalize(&workspace, normalized);

    print!(
        "out = {}",
        AtomPrinter::new(
            normalized.to_mul_view().to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get_buf().len());
}
