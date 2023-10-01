use symbolica::{
    representations::Atom,
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::new();

    let x = state.get_or_insert_var("x");
    let a = Atom::parse("f(x)+x^2*y", &mut state, &workspace).unwrap();

    let point = Atom::parse("1", &mut state, &workspace).unwrap();

    let mut out = workspace.new_atom();
    a.as_view()
        .taylor_series(x, point.as_view(), 3, &workspace, &state, &mut out);

    println!("{}", out.printer(&state));
}
