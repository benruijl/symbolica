use symbolica::{
    id::{Condition, Match, MatchSettings, PatternAtomTreeIterator},
    representations::Atom,
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::default();

    let expr = Atom::parse("f(z)*f(f(x))*f(y)", &mut state, &workspace).unwrap();
    let pat_expr = Atom::parse("f(x_)", &mut state, &workspace).unwrap();

    let pattern = pat_expr.as_view().into_pattern(&state);
    let restrictions = Condition::default();
    let settings = MatchSettings::default();

    println!(
        "> Matching pattern {} to {}:",
        pat_expr.printer(&state),
        expr.printer(&state)
    );

    let mut it =
        PatternAtomTreeIterator::new(&pattern, expr.as_view(), &state, &restrictions, &settings);
    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\tMatch at location {:?} - {:?}:", location, used_flags);
        for (id, v) in match_stack {
            print!("\t\t{} = ", state.get_name(*id));
            match v {
                Match::Single(s) => {
                    print!("{}", s.printer(&state))
                }
                Match::Multiple(slice_type, mm) => {
                    print!("{:?} ", slice_type);
                    for vv in mm {
                        print!("{}", vv.printer(&state));
                        print!(", ")
                    }
                }
                Match::FunctionName(f) => {
                    print!("Fn {}", state.get_name(*f))
                }
            }
            println!();
        }
    }
}
