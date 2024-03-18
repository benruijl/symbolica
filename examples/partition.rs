use symbolica::{id::Pattern, representations::Atom, state::State, transformer::Transformer};

fn main() {
    let input = Atom::parse("f(1,3,2,3,1)").unwrap();
    let f = State::get_symbol("f");
    let g = State::get_symbol("g");

    let o = Pattern::parse("f(x__)").unwrap().replace_all(
        input.as_view(),
        &Pattern::Transformer(Box::new((
            Some(Pattern::parse("x__").unwrap()),
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
