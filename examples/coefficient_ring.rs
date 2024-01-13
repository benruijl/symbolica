use std::sync::Arc;

use symbolica::{
    representations::Atom,
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::default();

    let expr = Atom::parse("x*z+x*(y+2)^-1*(y+z+1)", &mut state, &workspace).unwrap();
    println!("> In: {}", expr.printer(&state));

    let mut expr_yz = Atom::new();
    expr.as_view().set_coefficient_ring(
        &Arc::new(vec![
            state.get_or_insert_var("y").into(),
            state.get_or_insert_var("z").into(),
        ]),
        &state,
        &workspace,
        &mut expr_yz,
    );
    println!("> Coefficient ring y,z: {}", expr_yz.printer(&state));

    // the coefficient must downgrade from y,z to y
    let mut expr_y = Atom::new();
    expr_yz.as_view().set_coefficient_ring(
        &Arc::new(vec![state.get_or_insert_var("y").into()]),
        &state,
        &workspace,
        &mut expr_y,
    );
    println!("> Coefficient ring y: {}", expr_y.printer(&state));

    // the coefficient must downgrade from y,z to y
    let mut expr_exp = Atom::new();
    expr_y.as_view().expand(&workspace, &state, &mut expr_exp);
    println!(
        "> Coefficient ring y after expansion: {}",
        expr_exp.printer(&state)
    );
}
