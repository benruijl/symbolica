use symbolica::{representations::Atom, state::State};

fn main() {
    let mut state = State::get_global_state().write().unwrap();

    let input = Atom::parse("x*(1+a)+x*5*y+f(5,x)+2+y^2+x^2 + x^3", &mut state).unwrap();
    let x = state.get_or_insert_var("x");
    let key = state.get_or_insert_var("key");
    let coeff = state.get_or_insert_var("coeff");

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
            let f = out.to_fun(key);
            f.add_arg(a);
        })),
        Some(Box::new(move |a, out| {
            let f = out.to_fun(coeff);
            f.add_arg(a);
        })),
    );
    println!("\t{}", out);
}
