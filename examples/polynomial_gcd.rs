use symbolica::{
    atom::AtomCore,
    domains::{integer::Z, rational::Q},
    parse,
};
use tracing_subscriber::{fmt, prelude::*, util::SubscriberInitExt, EnvFilter};

fn gcd_integer_poly() {
    let a = parse!("5 + 8*x + 3*x^2 - 5*y - 3*x*y")
        .unwrap()
        .to_polynomial::<_, u8>(&Z, None);

    let b = parse!("5 + 5*x - 2*y + 3*x*y - 3*y^2")
        .unwrap()
        .to_polynomial::<_, u8>(&Z, a.variables.clone());

    println!("> Polynomial gcd of {} and {} =", a, b);
    println!("\t{}", a.gcd(&b));
}

fn gcd_rat_poly() {
    let a = parse!("3/4*x^2 + 3/4*x^3 + y + x*y + z + x*z")
        .unwrap()
        .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

    let b = parse!("3/2*x^2 + 2*y + 9/20*x^2*y + 3/5*y^2 + 2*z - 3/4*x^2*z - 2/5*y*z - z^2")
        .unwrap()
        .to_rational_polynomial::<_, _, u8>(&Q, &Z, a.get_variables().clone());

    println!("> Polynomial gcd of {} and {} =", a, b);
    println!("\t{}", a.gcd(&b));
}

fn main() {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_env("SYMBOLICA_LOG"))
        .init();

    gcd_integer_poly();
    gcd_rat_poly();
}
