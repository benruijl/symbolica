use symbolica::{
    atom::{Atom, AtomCore},
    domains::{integer::Z, rational_polynomial::RationalPolynomial},
};

fn main() {
    let expr = Atom::parse("(x*y^2*5+5)^2/(2*x+5)+(x+4)/(6*x^2+1)").unwrap();
    let rat: RationalPolynomial<_, u8> = expr.to_rational_polynomial(&Z, &Z, None);
    println!("{}", rat);
}
