use symbolica::{id::Pattern, representations::Atom};

fn main() {
    let expr = Atom::parse(" f(1,2,x) + f(1,2,3)").unwrap();
    let pat = Pattern::parse("f(1,2,y_)").unwrap();
    let rhs = Pattern::parse("f(1,2,y_+1)").unwrap();

    let out = pat.replace_all(expr.as_view(), &rhs, None, None);
    println!("{}", out);
}
