use symbolica::{
    representations::Atom,
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::get_global_state().write().unwrap();
    let workspace: Workspace = Workspace::default();

    let input = Atom::parse("(1+x)^3", &mut state, &workspace).unwrap();

    let mut o = Atom::new();
    input.as_view().expand(&workspace, &mut o);

    println!("> Expansion of {}: {}", input, o);
}
