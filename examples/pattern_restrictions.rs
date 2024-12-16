use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    coefficient::CoefficientView,
    domains::finite_field,
    id::{Condition, Match, MatchSettings, WildcardRestriction},
};
fn main() {
    let expr = Atom::parse("f(1,2,3,4,5,6,7)").unwrap();
    let pat_expr = Atom::parse("f(x__,y__,z__,w__)").unwrap();

    let pattern = pat_expr.as_view().to_pattern();

    let x = Symbol::new("x__");
    let y = Symbol::new("y__");
    let z = Symbol::new("z__");
    let w = Symbol::new("w__");

    let conditions = Condition::from((x, WildcardRestriction::Length(0, Some(2))))
        & (y, WildcardRestriction::Length(0, Some(4)))
        & (
            y,
            WildcardRestriction::Cmp(
                x,
                Box::new(|y, x| {
                    let len_x = match x {
                        Match::Multiple(_, s) => s.len(),
                        _ => 1,
                    };
                    let len_y = match y {
                        Match::Multiple(_, s) => s.len(),
                        _ => 1,
                    };
                    len_x >= len_y
                }),
            ),
        )
        & (
            z,
            WildcardRestriction::Filter(Box::new(|x: &Match| {
                if let Match::Single(AtomView::Num(num)) = x {
                    if let CoefficientView::Natural(x, y) = num.get_coeff_view() {
                        y == 1 && x > 0 && finite_field::is_prime_u64(x as u64)
                    } else {
                        false
                    }
                } else {
                    false
                }
            })),
        )
        & (w, WildcardRestriction::Length(0, None));
    let settings = MatchSettings::default();

    println!(
        "> Matching pattern {} : 0 <= len(x) <= 2, 0 <= len(y) <= 4, len(x) >= len(y) & is_prime(z) to {}:",
        pat_expr,
        expr
    );

    let mut it = expr.pattern_match(&pattern, Some(&conditions), Some(&settings));
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
