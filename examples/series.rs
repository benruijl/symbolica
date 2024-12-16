use symbolica::atom::{Atom, AtomCore, Symbol};

fn main() {
    let x = Symbol::new("x");
    let a = Atom::parse("(1-cos(x))/sin(x)+log(x+x^2+2)").unwrap();

    let out = a
        .series(x, Atom::new_num(0).as_view(), 4.into(), true)
        .unwrap();

    println!("{}", out);
}
