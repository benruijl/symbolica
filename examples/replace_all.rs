use ahash::HashMap;
use symbolica::{
    id::Pattern,
    parser::parse,
    printer::AtomPrinter,
    representations::{default::DefaultRepresentation, OwnedAtom},
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(z)*f(f(x))*h(f(3))")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let pat_expr = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pattern = Pattern::from_view(pat_expr.to_view(), &state);
    let restrictions = HashMap::default();

    let rhs_expr = parse("g(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let rhs = Pattern::from_view(rhs_expr.to_view(), &state);

    let mut out = OwnedAtom::new();

    pattern.replace_all(
        expr.to_view(),
        &rhs,
        &state,
        &workspace,
        &restrictions,
        &mut out,
    );

    println!(
        "> Replace all {}={} in {}: {}",
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
        AtomPrinter::new(out.to_view(), symbolica::printer::PrintMode::default(), &state)
    );
}
