use symbolica::{representations::Atom, state::State};

fn main() {
    let mut state = State::get_global_state().write().unwrap();

    let x = state.get_or_insert_var("x");
    let a = Atom::parse("f(x)+x^2*y", &mut state).unwrap();

    let point = Atom::parse("1", &mut state).unwrap();

    let out = a.as_view().taylor_series(x, point.as_view(), 3);

    println!("{}", out);
}
