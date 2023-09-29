use symbolica::{
    representations::{Atom, OwnedFun},
    state::{ResettableBuffer, State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::default();

    let input = Atom::parse(
        "x*(1+a)+x*5*y+f(5,x)+2+y^2+x^2 + x^3",
        &mut state,
        &workspace,
    )
    .unwrap();
    let x = state.get_or_insert_var("x");
    let key = state.get_or_insert_var("key");
    let coeff = state.get_or_insert_var("coeff");

    let (r, rest) = input.as_view().coefficient_list(x, &workspace, &state);

    println!("> Coefficient list:");
    for (key, val) in r {
        println!("\t{} {}", key.printer(&state), val.printer(&state));
    }
    println!("\t1 {}", rest.printer(&state));

    println!("> Collect in x:");
    let mut out = Atom::new();
    input.as_view().collect(
        x,
        &workspace,
        &state,
        Some(Box::new(|x, out| {
            out.set_from_view(&x);
        })),
        None,
        &mut out,
    );
    println!("\t{}", out.printer(&state));

    println!("> Collect in x with wrapping:");
    let mut out = Atom::new();
    input.as_view().collect(
        x,
        &workspace,
        &state,
        Some(Box::new(move |a, out| {
            let f = out.to_fun();
            f.set_from_name(key);
            f.add_arg(a);
        })),
        Some(Box::new(move |a, out| {
            let f = out.to_fun();
            f.set_from_name(coeff);
            f.add_arg(a);
        })),
        &mut out,
    );
    println!("\t{}", out.printer(&state));
}
