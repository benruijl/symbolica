use symbolica::{id::Pattern, representations::Atom, state::State};

fn main() {
    let mut state = State::get_global_state().write().unwrap();

    let expr = Atom::parse(" f(1,2,x) + f(1,2,3)", &mut state).unwrap();
    let pat = Pattern::parse("f(1,2,y_)", &mut state).unwrap();
    let rhs = Pattern::parse("f(1,2,y_+1)", &mut state).unwrap();

    let out = pat.replace_all(expr.as_view(), &rhs, None, None);
    println!("{}", out);
}
