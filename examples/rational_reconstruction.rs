use symbolica::domains::{
    finite_field::{FiniteFieldCore, Zp},
    rational::{Rational, Q},
    Field, Ring,
};

/// An arbitrarily complicated black-box function.
fn black_box(field: &Zp, eval: &[<Zp as Ring>::Element]) -> <Zp as Ring>::Element {
    // eval -(x^2+y/3)
    field.neg(&field.add(
        &field.mul(&eval[0], &eval[0]),
        &field.div(&eval[1], &field.to_element(3)),
    ))
}

fn main() {
    let r = Rational::rational_reconstruction::<_, Q>(
        black_box,
        &[Rational::Natural(1, 2), Rational::Natural(3, 1)],
        None,
    );

    assert_eq!(r, Ok(Rational::Natural(-5, 4)));
    println!("Reconstructed f(1/2,3)={}", r.unwrap());
}
