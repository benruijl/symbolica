use symbolica::{
    atom::{Atom, AtomCore},
    id::Pattern,
};

fn main() {
    let expr = Atom::parse(" f(1,2,x) + f(1,2,3)").unwrap();
    let pat = Pattern::parse("f(1,2,y_)").unwrap();
    let rhs = Pattern::parse("f(1,2,y_+1)").unwrap();

    let out = expr.replace_all(&pat, &rhs, None, None);
    println!("{}", out);
}
