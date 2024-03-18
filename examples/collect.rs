use symbolica::{
    fun,
    representations::{Atom, FunctionBuilder},
    state::State,
};

fn main() {
    let input = Atom::parse("x*(1+a)+x*5*y+f(5,x)+2+y^2+x^2 + x^3").unwrap();
    let x = State::get_symbol("x");
    let key = State::get_symbol("key");
    let coeff = State::get_symbol("coeff");

    let (r, rest) = input.coefficient_list(x);

    println!("> Coefficient list:");
    for (key, val) in r {
        println!("\t{} {}", key, val);
    }
    println!("\t1 {}", rest);

    println!("> Collect in x:");
    let out = input.collect(
        x,
        Some(Box::new(|x, out| {
            out.set_from_view(&x);
        })),
        None,
    );
    println!("\t{}", out);

    println!("> Collect in x with wrapping:");
    let out = input.collect(
        x,
        Some(Box::new(move |a, out| {
            out.set_from_view(&a);
            *out = fun!(key, out);
        })),
        Some(Box::new(move |a, out| {
            out.set_from_view(&a);
            *out = fun!(coeff, out);
        })),
    );
    println!("\t{}", out);
}
