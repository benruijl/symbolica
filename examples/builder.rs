use symbolica::{
    fun,
    representations::{Atom, FunctionBuilder},
    state::State,
};

fn main() {
    let mut state = State::get_global_state().write().unwrap();

    let x = Atom::parse("x", &mut state).unwrap();
    let y = Atom::parse("y", &mut state).unwrap();
    let f_id = state.get_or_insert_fn("f", None).unwrap();

    let f = fun!(f_id, x, y, Atom::new_num(2));

    let xb = (-(&y + &x + 2) * &y * 6).npow(5) / &y * &f / 4;

    println!("{}", xb);
}
