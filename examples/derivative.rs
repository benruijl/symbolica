use symbolica::{
    parser::parse,
    printer::{self, AtomPrinter},
    representations::{default::DefaultRepresentation, OwnedAtom},
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace<DefaultRepresentation> = Workspace::new();

    let inputs = [
        "(1+2*x)^(5+x)",
        "log(2*x) + exp(3*x) + sin(4*x) + cos(y*x)",
        "f(x^2,x)",
        "der(0,1,f(x,x^3))",
    ];

    for input in inputs {
        let input = parse(input)
            .unwrap()
            .to_atom(&mut state, &workspace)
            .unwrap();

        let mut a = OwnedAtom::new();
        input
            .to_view()
            .derivative(state.get_or_insert_var("x"), &workspace, &state, &mut a);

        println!(
            "d({})/dx = {}:",
            AtomPrinter::new(input.to_view(), printer::PrintMode::default(), &state),
            AtomPrinter::new(a.to_view(), printer::PrintMode::default(), &state)
        );
    }
}
