use symbolica::representations::{
    default::OwnedAtom,
    tree::{Atom, Number},
    OwnedAtomT,
};

fn main() {
    let a = Atom::Term(vec![
        Atom::Var(15, Number::new(2, 1)),
        Atom::Number(Number::new(2000, 1)),
        Atom::Var(16, Number::new(2, 1)),
        Atom::Var(15, Number::new(1, 2)),
        Atom::Number(Number::new(4, 4000)),
        Atom::Fn(17, vec![Atom::Var(15, Number::new(3, 4))]),
    ]);
    println!("expr={:?} ,len = {} bytes", a, a.len());

    let b = OwnedAtom::from_tree(&a);

    println!("lin size: {:?} bytes", b.len());

    assert!(
        a == b.to_tree(),
        "Not similar: {:?} vs {:?}",
        a,
        b.to_tree()
    );

    let view = b.to_view();

    view.print_tree(0);

    let normalized = view.normalize();

    normalized.to_view().print_tree(0);

    normalized.to_view().print();
}
