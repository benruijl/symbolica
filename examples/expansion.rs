use symbolica::{
    parser::parse,
    printer::AtomPrinter,
    representations::{default::DefaultRepresentation, OwnedAtom},
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace<DefaultRepresentation> = Workspace::new();

    let input = parse("(1+y+x)^3*(x+1)*y+5")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let mut o = OwnedAtom::new();

    input.to_view().expand(&workspace, &state, &mut o);

    println!(
        "> Expansion of {}: {}",
        AtomPrinter::new(
            input.to_view(),
            symbolica::printer::PrintMode::default(),
            &state
        ),
        AtomPrinter::new(
            o.to_view(),
            symbolica::printer::PrintMode::default(),
            &state
        ),
    );
}
