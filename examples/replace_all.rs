use symbolica::{
    id::Pattern,
    representations::Atom,
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::get_global_state().write().unwrap();
    let workspace = Workspace::default();

    let expr = Atom::parse(" f(1,2,x) + f(1,2,3)", &mut state, &workspace).unwrap();
    let pat = Pattern::parse("f(1,2,y_)", &mut state, &workspace).unwrap();
    let rhs = Pattern::parse("f(1,2,y_+1)", &mut state, &workspace).unwrap();

    let mut out = Atom::new();
    pat.replace_all(expr.as_view(), &rhs, &workspace, None, None, &mut out);
    println!("{}", out);
}
