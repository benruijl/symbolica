use ahash::HashMap;
use symbolica::{
    id::Pattern,
    representations::Atom,
    state::{ResettableBuffer, State, Workspace},
    transformer::Transformer,
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::default();

    let input = Atom::parse("f(1,3,2,3,1)", &mut state, &workspace).unwrap();
    let f = state.get_or_insert_fn("f", None).unwrap();
    let g = state.get_or_insert_fn("g", None).unwrap();

    let mut o = Atom::new();
    Pattern::parse("f(x_)", &mut state, &workspace)
        .unwrap()
        .replace_all(
            input.as_view(),
            &Pattern::Transformer(Box::new(Transformer::Partition(
                Pattern::parse("x_", &mut state, &workspace).unwrap(),
                vec![(f, 2), (g, 2), (f, 1)],
                false,
                false,
            ))),
            &state,
            &workspace,
            &HashMap::default(),
            &mut o,
        );

    println!("> {}", o.printer(&state));
}
