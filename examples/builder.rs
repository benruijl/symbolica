use symbolica::{
    fun,
    representations::{Atom, FunctionBuilder},
    state::State,
};

fn main() {
    let x = Atom::parse("x").unwrap();
    let y = Atom::parse("y").unwrap();
    let f_id = State::get_symbol("f");

    let f = fun!(f_id, x, y, Atom::new_num(2));

    let xb = (-(&y + &x + 2) * &y * 6).npow(5) / &y * &f / 4;

    println!("{}", xb);
}
