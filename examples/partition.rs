use symbolica::{id::Pattern, representations::Atom, state::State, transformer::Transformer};

fn main() {
    let mut state = State::get_global_state().write().unwrap();

    let input = Atom::parse("f(1,3,2,3,1)", &mut state).unwrap();
    let f = state.get_or_insert_fn("f", None).unwrap();
    let g = state.get_or_insert_fn("g", None).unwrap();

    let o = Pattern::parse("f(x__)", &mut state).unwrap().replace_all(
        input.as_view(),
        &Pattern::Transformer(Box::new((
            Some(Pattern::parse("x__", &mut state).unwrap()),
            vec![Transformer::Partition(
                vec![(f, 2), (g, 2), (f, 1)],
                false,
                false,
            )],
        ))),
        None,
        None,
    );

    println!("> {}", o);
}
