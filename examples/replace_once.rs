use ahash::HashMap;
use symbolica::{
    id::{Pattern, ReplaceIterator},
    parser::parse,
    printer::AtomPrinter,
    representations::{default::DefaultRepresentation, OwnedAtom},
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(z)*f(f(x))*f(y)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let pat_expr = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let rhs_expr = parse("g(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let rhs = Pattern::from_view(rhs_expr.to_view(), &state);

    let pattern = Pattern::from_view(pat_expr.to_view(), &state);
    let restrictions = HashMap::default();

    println!(
        "> Replace once {}={} in {}:",
        AtomPrinter::new(
            pat_expr.to_view(),
            symbolica::printer::PrintMode::default(),
            &state
        ),
        AtomPrinter::new(
            rhs_expr.to_view(),
            symbolica::printer::PrintMode::default(),
            &state
        ),
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::default(), &state),
    );

    let mut replaced = OwnedAtom::new();

    let mut it = ReplaceIterator::new(&pattern, expr.to_view(), &rhs, &state, &restrictions);
    while let Some(()) = it.next(&workspace, &mut replaced) {
        println!(
            "\t{}",
            AtomPrinter::new(
                replaced.to_view(),
                symbolica::printer::PrintMode::default(),
                &state
            ),
        );
    }
}
