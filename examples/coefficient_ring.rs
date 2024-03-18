use std::sync::Arc;

use symbolica::{representations::Atom, state::State};

fn main() {
    let expr = Atom::parse("x*z+x*(y+2)^-1*(y+z+1)").unwrap();
    println!("> In: {}", expr);

    let expr_yz = expr.set_coefficient_ring(&Arc::new(vec![
        State::get_symbol("y").into(),
        State::get_symbol("z").into(),
    ]));
    println!("> Coefficient ring y,z: {}", expr_yz);

    // the coefficient must downgrade from y,z to y
    let expr_y = expr_yz.set_coefficient_ring(&Arc::new(vec![State::get_symbol("y").into()]));
    println!("> Coefficient ring y: {}", expr_y);

    // the coefficient must downgrade from y,z to y
    let expr_exp = expr_y.expand();
    println!("> Coefficient ring y after expansion: {}", expr_exp);
}
