use symbolica::{
    representations::{AsAtomView, Atom, AtomBuilder},
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let ws: Workspace = Workspace::new();

    let x = Atom::parse("x", &mut state, &ws).unwrap();
    let y = Atom::parse("y", &mut state, &ws).unwrap();

    // instead of parsing an expression, we build it from components
    let mut e = Atom::new();
    x.add(&state, &ws, &y, &mut e);
    println!("{}", e.printer(&state));

    // the cumbersome passing of the state and workspace can be avoided by using an
    // AtomBuilder, which accumulates the result
    let mut res = Atom::new();
    let mut xb = AtomBuilder::new(&x, &state, &ws, &mut res);

    xb = (-(xb + &y + &x) * &y * &ws.new_num(6)).pow(&ws.new_num(5)) / &y;

    println!("{}", xb.to_atom_mut().printer(&state));
}
