use symbolica::{
    representations::{AsAtomView, Atom, FunctionBuilder},
    state::{FunctionAttribute, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let ws: Workspace = Workspace::new();

    let x = Atom::parse("x", &mut state, &ws).unwrap();
    let y = Atom::parse("y", &mut state, &ws).unwrap();
    let f_id = state
        .get_or_insert_fn("f", Some(vec![FunctionAttribute::Symmetric]))
        .unwrap();
    let f = FunctionBuilder::new(f_id, &state, &ws)
        .add_arg(&ws.new_num(1))
        .finish();

    // the cumbersome passing of the state and workspace can be avoided by using an
    // AtomBuilder, which accumulates the result
    let mut xb = x.builder(&state, &ws);

    xb = (-(xb + &y + &x) * &y * &ws.new_num(6)).pow(&ws.new_num(5)) / &y * &f;

    println!("{}", xb.as_atom_view().printer(&state));
}
