use symbolica::{representations::Atom, state::State};

fn main() {
    let mut state = State::get_global_state().write().unwrap();

    let input = Atom::parse("(1+x)^3", &mut state).unwrap();

    let o = input.expand();

    println!("> Expansion of {}: {}", input, o);
}
