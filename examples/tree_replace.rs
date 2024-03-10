use symbolica::{
    id::{Condition, Match, MatchSettings, PatternAtomTreeIterator},
    representations::Atom,
    state::State,
};

fn main() {
    let expr = Atom::parse("f(z)*f(f(x))*f(y)").unwrap();
    let pat_expr = Atom::parse("f(x_)").unwrap();

    let pattern = pat_expr.as_view().into_pattern();
    let restrictions = Condition::default();
    let settings = MatchSettings::default();

    println!("> Matching pattern {} to {}:", pat_expr, expr);

    let mut it = PatternAtomTreeIterator::new(&pattern, expr.as_view(), &restrictions, &settings);
    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\tMatch at location {:?} - {:?}:", location, used_flags);
        for (id, v) in match_stack {
            print!("\t\t{} = ", State::get_name(*id));
            match v {
                Match::Single(s) => {
                    print!("{}", s)
                }
                Match::Multiple(slice_type, mm) => {
                    print!("{:?} ", slice_type);
                    for vv in mm {
                        print!("{}", vv);
                        print!(", ")
                    }
                }
                Match::FunctionName(f) => {
                    print!("Fn {}", State::get_name(*f))
                }
            }
            println!();
        }
    }
}
