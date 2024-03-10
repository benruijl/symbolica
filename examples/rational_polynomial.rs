use symbolica::{
    domains::{
        integer::IntegerRing, rational::RationalField, rational_polynomial::RationalPolynomial,
    },
    representations::Atom,
    state::Workspace,
};

fn main() {
    let workspace = Workspace::new();

    let expr = Atom::parse("(x*y^2*5+5)^2/(2*x+5)+(x+4)/(6*x^2+1)").unwrap();
    let rat: RationalPolynomial<IntegerRing, u8> = expr
        .as_view()
        .to_rational_polynomial(&workspace, &RationalField::new(), &IntegerRing::new(), None)
        .unwrap();
    println!("{}", rat);
}
