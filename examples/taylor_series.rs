use symbolica::{representations::Atom, state::State};

fn main() {
    let x = State::get_symbol("x");
    let a = Atom::parse("f(x)+x^2*y").unwrap();

    let point = Atom::parse("1").unwrap();

    let out = a.taylor_series(x, point.as_view(), 3);

    println!("{}", out);
}
