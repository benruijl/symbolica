use symbolica::{
    parser::parse,
    printer::{AtomPrinter, PrintMode},
    representations::{default::DefaultRepresentation, OwnedAtom},
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("x*z+x*(y+2)^-1*(y+z+1)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    println!(
        "> In: {}",
        AtomPrinter::new(expr.to_view(), PrintMode::default(), &state)
    );

    let mut expr_yz = OwnedAtom::new();
    expr.to_view().set_coefficient_ring(
        &[state.get_or_insert_var("y"), state.get_or_insert_var("z")],
        &state,
        &workspace,
        &mut expr_yz,
    );
    println!(
        "> Coefficient ring y,z: {}",
        AtomPrinter::new(expr_yz.to_view(), PrintMode::default(), &state)
    );

    // the coefficient must downgrade from y,z to y
    let mut expr_y = OwnedAtom::new();
    expr_yz.to_view().set_coefficient_ring(
        &[state.get_or_insert_var("y")],
        &state,
        &workspace,
        &mut expr_y,
    );
    println!(
        "> Coefficient ring y: {}",
        AtomPrinter::new(expr_y.to_view(), PrintMode::default(), &state)
    );

    // the coefficient must downgrade from y,z to y
    let mut expr_exp = OwnedAtom::new();
    expr_y
        .to_view()
        .expand(&workspace, &mut state, &mut expr_exp);
    println!(
        "> Coefficient ring y after expansion: {}",
        AtomPrinter::new(expr_exp.to_view(), PrintMode::default(), &state)
    );
}
