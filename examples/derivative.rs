use symbolica::{
    representations::Atom,
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::default();

    let inputs = [
        "(1+2*x)^(5+x)",
        "log(2*x) + exp(3*x) + sin(4*x) + cos(y*x)",
        "f(x^2,x)",
        "der(0,1,f(x,x^3))",
    ];

    for input in inputs {
        let input = Atom::parse(input, &mut state, &workspace).unwrap();

        let mut a = Atom::new();
        input
            .as_view()
            .derivative(state.get_or_insert_var("x"), &workspace, &state, &mut a);

        println!("d({})/dx = {}:", input.printer(&state), a.printer(&state));
    }
}
