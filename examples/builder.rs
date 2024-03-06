use symbolica::{
    representations::{Atom, FunctionBuilder},
    state::{FunctionAttribute, RecycledAtom, State},
};

fn main() {
    let mut state = State::get_global_state().write().unwrap();

    let x = Atom::parse("x", &mut state).unwrap();
    let y = Atom::parse("y", &mut state).unwrap();
    let f_id = state
        .get_or_insert_fn("f", Some(vec![FunctionAttribute::Symmetric]))
        .unwrap();
    let f = FunctionBuilder::new(f_id)
        .add_arg(&RecycledAtom::new_num(6))
        .add_arg(&x)
        .finish();

    let mut xb = x.clone();
    xb = (-(xb + &y + &x) * &y * &RecycledAtom::new_num(6)).pow(&Atom::new_num(5)) / &y * &f;

    println!("{}", xb);
}
