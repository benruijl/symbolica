use symbolica::{
    atom::{Atom, AtomCore},
    id::Match,
};

fn main() {
    let expr = Atom::parse("x*y*w*z*f(x,y,x*y,z)").unwrap();

    let pat_expr = Atom::parse("z*x_*y___*g___(z___,x_,w___)").unwrap();

    let pattern = pat_expr.to_pattern();

    println!("> Matching pattern {} to {}:", pat_expr, expr.as_view());

    // simple match
    for m in expr.pattern_match(&pattern, None, None) {
        for (wc, v) in m {
            println!("\t{} = {}", wc, v);
        }
        println!();
    }

    // advanced match
    let mut it = expr.pattern_match(&pattern, None, None);
    while let Some(m) = it.next_detailed() {
        println!(
            "\t Match at location {:?} - {:?}:",
            m.position, m.used_flags
        );
        for (id, v) in m.match_stack {
            print!("\t\t{} = ", id);
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
                    print!("Fn {}", f)
                }
            }
            println!();
        }
    }
}
