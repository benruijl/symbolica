use symbolica::{
    id::{Condition, MatchSettings},
    representations::Atom,
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::default();

    let expr = Atom::parse("f(z)*f(f(x))*f(y)", &mut state, &workspace).unwrap();
    let pat_expr = Atom::parse("f(x_)", &mut state, &workspace).unwrap();

    let rhs_expr = Atom::parse("g(x_)", &mut state, &workspace).unwrap();
    let rhs = rhs_expr.as_view().into_pattern(&state);

    let pattern = pat_expr.as_view().into_pattern(&state);
    let restrictions = Condition::default();
    let settings = MatchSettings::default();

    println!(
        "> Replace once {}={} in {}:",
        pat_expr.as_view().printer(&state),
        rhs_expr.as_view().printer(&state),
        expr.as_view().printer(&state)
    );

    let mut replaced = Atom::new();

    let mut it = pattern.replace_iter(expr.as_view(), &rhs, &state, &restrictions, &settings);
    while let Some(()) = it.next(&state, &workspace, &mut replaced) {
        println!("\t{}", replaced.printer(&state));
    }
}
