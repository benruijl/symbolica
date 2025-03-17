use symbolica::{atom::AtomCore, id::Pattern, parse, symbol, transformer::Transformer};

fn main() {
    let input = parse!("f(1,3,2,3,1)").unwrap();
    let (f, g) = symbol!("f", "g");

    let o = input
        .replace(parse!("f(x__)").unwrap())
        .with(Pattern::Transformer(Box::new((
            Some(parse!("x__").unwrap().to_pattern()),
            vec![Transformer::Partition(
                vec![(f, 2), (g, 2), (f, 1)],
                false,
                false,
            )],
        ))));

    println!("> {}", o);
}
