use symbolica::{atom::AtomCore, parse};

fn main() {
    let expr = parse!(" f(1,2,x) + f(1,2,3)").unwrap();
    let pat = parse!("f(1,2,y_)").unwrap();
    let rhs = parse!("f(1,2,y_+1)").unwrap();

    let out = expr.replace(pat).with(rhs);
    println!("{}", out);
}
