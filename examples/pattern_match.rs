use ahash::HashMap;
use symbolica::{
    id::{Match, PatternRestriction},
    representations::Atom,
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::default();

    let expr = Atom::parse("x*y*w*z*f(x,y,x*y,z)", &mut state, &workspace).unwrap();

    let pat_expr = Atom::parse("z*x_*y_*g_(z_,x_,w_)", &mut state, &workspace).unwrap();

    let pattern = pat_expr.as_view().into_pattern(&state);
    let mut restrictions = HashMap::default();
    restrictions.insert(
        state.get_or_insert_var("x_"),
        vec![PatternRestriction::Length(1, Some(100))],
    );

    println!(
        "> Matching pattern {} to {}:",
        pat_expr.printer(&state),
        expr.as_view().printer(&state)
    );

    let mut it = pattern.pattern_match(expr.as_view(), &state, &restrictions);
    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\t Match at location {:?} - {:?}:", location, used_flags);
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
