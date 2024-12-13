use symbolica::{
    atom::{Atom, AtomCore},
    id::{Condition, Match, MatchSettings, PatternAtomTreeIterator},
    state::State,
};

fn main() {
    let expr = Atom::parse("f(z)*f(f(x))*f(y)").unwrap();
    let pat_expr = Atom::parse("f(x_)").unwrap();

    let pattern = pat_expr.to_pattern();
    let restrictions = Condition::default();
    let settings = MatchSettings::default();

    println!("> Matching pattern {} to {}:", pat_expr, expr);

    let mut it = PatternAtomTreeIterator::new(&pattern, expr.as_view(), &restrictions, &settings);
    while let Some(m) = it.next() {
        println!("\tMatch at location {:?} - {:?}:", m.position, m.used_flags);
        for (id, v) in m.match_stack {
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
