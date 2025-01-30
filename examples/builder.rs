use symbolica::{fun, symbol};

fn main() {
    let (x, y, f) = symbol!("x", "y", "f");

    let f = fun!(f, x, y, 2);

    let xb = (-(y + x + 2) * y * 6).npow(5) / y * f / 4;

    println!("{}", xb);
}
