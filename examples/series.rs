use symbolica::atom::{Atom, AtomCore, Symbol};

fn main() {
    let x = Symbol::new("x");
    let a = Atom::parse("(1-cos(x))/sin(x)").unwrap();

    let out = a.series(x, Atom::new_num(0), 4.into(), true).unwrap();

    println!("{}", out);
}
