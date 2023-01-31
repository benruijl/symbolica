use rug::Rational;
use symbolica::{
    finite_field::{FiniteFieldU64, PrimeIteratorU64},
    parser::parse,
    printer::AtomPrinter,
    representations::{number::Number, tree::AtomTree, OwnedAtom},
    state::{ResettableBuffer, State, Workspace},
};

fn expression_test() {
    let mut state = State::new();

    // create variable names
    let (x, y, z) = (
        state.get_or_insert_var("x"),
        state.get_or_insert_var("y"),
        state.get_or_insert_var("z"),
    );

    // create term
    let a = AtomTree::Mul(vec![
        AtomTree::Pow(Box::new((
            AtomTree::Var(x),
            AtomTree::Num(Number::Natural(2, 1)),
        ))),
        AtomTree::Num(Number::Natural(3, 1)),
        AtomTree::Pow(Box::new((
            AtomTree::Var(x),
            AtomTree::Num(Number::Natural(-2, 1)),
        ))),
        AtomTree::Var(y),
        AtomTree::Num(Number::Large(
            Rational::from_str_radix(
                "1723671261273182378912738921/128937127893761293712893712983712",
                10,
            )
            .unwrap(),
        )),
        AtomTree::Var(y),
        AtomTree::Fn(
            z,
            vec![AtomTree::Add(vec![
                AtomTree::Num(Number::Natural(1, 1)),
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

    let normalized = normalized_handle.get_buf_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "out = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get_buf().len());
}

fn finite_field_test() {
    let mut state = State::new();

    let prime = PrimeIteratorU64::new(16).next().unwrap();
    let f = FiniteFieldU64::new(prime);
    let fi = state.get_or_insert_finite_field(f.clone());

    let x = state.get_or_insert_var("x");

    // create term x * (13 % 17) * (15 % 17)
    let a = AtomTree::Mul(vec![
        AtomTree::Num(Number::FiniteField(f.to_montgomery(13), fi)),
        AtomTree::Num(Number::FiniteField(f.to_montgomery(15), fi)),
        AtomTree::Var(x),
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

    let normalized = normalized_handle.get_buf_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "out = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get_buf().len());
}

fn parse_test() {
    let mut state = State::new();

    // spaces and underscores are allowed in numbers are are all stripped
    let token = parse("(1+  x^2/5  )*443_555*f(\t2*1,\n4* 44 5 + \r\n 2)^5\\*6").unwrap();

    let a = token.to_atom_tree(&mut state).unwrap();

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
    let normalized = normalized_handle.get_buf_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "out = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get_buf().len());
}

fn main() {
    expression_test();
    finite_field_test();
    parse_test();
}
