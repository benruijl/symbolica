use ahash::HashMap;
use symbolica::{
    id::{Match, Pattern, PatternAtomTreeIterator},
    parser::parse,
    printer::AtomPrinter,
    representations::{default::DefaultRepresentation, OwnedAtom},
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(z)*f(f(x))*f(y)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let pat_expr = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pattern = Pattern::from_view(pat_expr.to_view(), &state);
    let restrictions = HashMap::default();

    println!(
        "> Matching pattern {} to {}:",
        AtomPrinter::new(
            pat_expr.to_view(),
            symbolica::printer::PrintMode::default(),
            &state
        ),
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::default(), &state)
    );

    let mut it = PatternAtomTreeIterator::new(&pattern, expr.to_view(), &state, &restrictions);
    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\tMatch at location {:?} - {:?}:", location, used_flags);
        for (id, v) in match_stack {
            print!("\t\t{} = ", state.get_name(*id).unwrap());
            match v {
                Match::Single(s) => print!(
                    "{}",
                    AtomPrinter::new(*s, symbolica::printer::PrintMode::default(), &state),
                ),
                Match::Multiple(slice_type, mm) => {
                    print!("{:?} ", slice_type);
                    for vv in mm {
                        print!(
                            "{}",
                            AtomPrinter::new(*vv, symbolica::printer::PrintMode::default(), &state),
                        );
                        print!(", ")
                    }
                }
                Match::FunctionName(f) => {
                    print!("Fn {}", state.get_name(*f).unwrap())
                }
            }
            println!("");
        }
    }
}
