use symbolica::{atom::AtomCore, parse};

fn main() {
    let expr = parse!(" f(1,2,x) + f(1,2,3)").unwrap();
    let pat = parse!("f(1,2,y_)").unwrap().to_pattern();
    let rhs = parse!("f(1,2,y_+1)").unwrap().to_pattern();

    let out = expr.replace_all(&pat, &rhs, None, None);
    println!("{}", out);
}
