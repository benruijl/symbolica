use symbolica::{
    atom::{Atom, AtomCore},
    id::Match,
    symb,
};

fn main() {
    let expr = Atom::parse("f(z)*f(f(x))*f(y)").unwrap();
    let pat_expr = Atom::parse("f(x_)").unwrap();

    let pattern = pat_expr.to_pattern();

    println!("> Matching pattern {} to {}:", pat_expr, expr);

    for x in expr.pattern_match(&pattern, None, None) {
        println!("\t x_ = {}", x.get(&symb!("x_")).unwrap());
    }

    println!("> Matching pattern {} to {}:", pat_expr, expr);

    // use next_detailed for detailed information
    let mut it = expr.pattern_match(&pattern, None, None);
    while let Some(m) = it.next_detailed() {
        println!("\tMatch at location {:?} - {:?}:", m.position, m.used_flags);
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
