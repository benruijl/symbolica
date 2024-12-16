use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    id::Pattern,
    transformer::Transformer,
};

fn main() {
    let input = Atom::parse("f(1,3,2,3,1)").unwrap();
    let f = Symbol::new("f");
    let g = Symbol::new("g");

    let o = input.replace_all(
        &Pattern::parse("f(x__)").unwrap(),
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
