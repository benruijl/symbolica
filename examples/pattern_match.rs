use symbolica::{
    id::{Condition, Match, MatchSettings},
    representations::Atom,
    state::State,
};

fn main() {
    let expr = Atom::parse("x*y*w*z*f(x,y,x*y,z)").unwrap();

    let pat_expr = Atom::parse("z*x_*y___*g___(z___,x_,w___)").unwrap();

    let pattern = pat_expr.as_view().into_pattern();
    let conditions = Condition::default();
    let settings = MatchSettings::default();

    println!("> Matching pattern {} to {}:", pat_expr, expr.as_view());

    let mut it = pattern.pattern_match(expr.as_view(), &conditions, &settings);
    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\t Match at location {:?} - {:?}:", location, used_flags);
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
