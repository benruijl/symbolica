use symbolica::{
    id::{Match, MatchSettings, PatternRestriction},
    representations::Atom,
    state::State,
};

fn main() {
    let expr = Atom::parse("x*y*w*z*f(x,y,x*y,z)").unwrap();

    let pat_expr = Atom::parse("z*x_*y_*g_(z_,x_,w_)").unwrap();

    let pattern = pat_expr.as_view().into_pattern();
    let restrictions = (
        State::get_or_insert_var("x_"),
        PatternRestriction::Length(1, Some(100)),
    )
        .into();
    let settings = MatchSettings::default();

    println!("> Matching pattern {} to {}:", pat_expr, expr.as_view());

    let mut it = pattern.pattern_match(expr.as_view(), &restrictions, &settings);
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
