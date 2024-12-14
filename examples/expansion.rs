use symbolica::atom::Atom;

fn main() {
    let input = Atom::parse("(1+x)^3").unwrap();

    let o = input.expand();

    println!("> Expansion of {}: {}", input, o);
}
