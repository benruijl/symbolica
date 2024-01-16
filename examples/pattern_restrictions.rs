use ahash::HashMap;
use symbolica::{
    id::{Match, PatternRestriction},
    representations::{number::BorrowedNumber, Atom, AtomView, Num},
    domains::finite_field,
    state::{State, Workspace},
};
fn main() {
    let mut state = State::new();
    let workspace = Workspace::default();

    let expr = Atom::parse("f(1,2,3,4,5,6,7)", &mut state, &workspace).unwrap();
    let pat_expr = Atom::parse("f(x_,y_,z_,w_)", &mut state, &workspace).unwrap();

    let pattern = pat_expr.as_view().into_pattern(&state);
    let mut restrictions = HashMap::default();
    restrictions.insert(
        state.get_or_insert_var("x_"),
        vec![PatternRestriction::Length(0, Some(2))],
    );
    restrictions.insert(
        state.get_or_insert_var("y_"),
        vec![
            PatternRestriction::Length(0, Some(4)),
            PatternRestriction::Cmp(
                state.get_or_insert_var("x_"),
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
        ],
    );
    restrictions.insert(
        state.get_or_insert_var("z_"),
        vec![PatternRestriction::Filter(Box::new(|x: &Match| {
            if let Match::Single(AtomView::Num(num)) = x {
                if let BorrowedNumber::Natural(x, y) = num.get_number_view() {
                    y == 1 && x > 0 && finite_field::is_prime_u64(x as u64)
                } else {
                    false
                }
            } else {
                false
            }
        }))],
    );
    restrictions.insert(
        state.get_or_insert_var("w_"),
        vec![PatternRestriction::Length(0, None)],
    );

    println!(
        "> Matching pattern {} : 0 <= len(x) <= 2, 0 <= len(y) <= 4, len(x) >= len(y) & is_prime(z) to {}:",
        pat_expr.printer(&state),
        expr.printer(&state)
    );

    let mut it = pattern.pattern_match(expr.as_view(), &state, &restrictions);
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
