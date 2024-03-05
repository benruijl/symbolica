use symbolica::{
    id::{Condition, MatchSettings},
    representations::Atom,
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::get_global_state().write().unwrap();
    let workspace = Workspace::default();

    let expr = Atom::parse("f(z)*f(f(x))*f(y)", &mut state, &workspace).unwrap();
    let pat_expr = Atom::parse("f(x_)", &mut state, &workspace).unwrap();

    let rhs_expr = Atom::parse("g(x_)", &mut state, &workspace).unwrap();
    let rhs = rhs_expr.as_view().into_pattern();

    let pattern = pat_expr.as_view().into_pattern();
    let restrictions = Condition::default();
    let settings = MatchSettings::default();

    println!(
        "> Replace once {}={} in {}:",
        pat_expr.as_view(),
        rhs_expr.as_view(),
        expr.as_view()
    );

    let mut replaced = Atom::new();

    let mut it = pattern.replace_iter(expr.as_view(), &rhs, &restrictions, &settings);
    while let Some(()) = it.next(&workspace, &mut replaced) {
        println!("\t{}", replaced);
    }
}
