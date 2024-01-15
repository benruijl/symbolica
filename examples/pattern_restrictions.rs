use ahash::HashMap;
use symbolica::{
    domains::finite_field,
    id::{Condition, Match, PatternRestriction},
    representations::{number::BorrowedNumber, Atom, AtomView, Num},
    state::{State, Workspace},
};
fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::default();

    let expr = Atom::parse("f(1,2,3,4,5,6,7)", &mut state, &workspace).unwrap();
    let pat_expr = Atom::parse("f(x__,y__,z__,w__)", &mut state, &workspace).unwrap();

    let pattern = pat_expr.as_view().into_pattern(&state);

    let x = state.get_or_insert_var("x__");
    let y = state.get_or_insert_var("y__");
    let z = state.get_or_insert_var("z__");
    let w = state.get_or_insert_var("w__");

    let conditions = Condition::from((x, PatternRestriction::Length(0, Some(2))))
        & (y, PatternRestriction::Length(0, Some(4)))
        & (
            y,
            PatternRestriction::Cmp(
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
            PatternRestriction::Filter(Box::new(|x: &Match| {
                if let Match::Single(AtomView::Num(num)) = x {
                    if let BorrowedNumber::Natural(x, y) = num.get_number_view() {
                        y == 1 && x > 0 && finite_field::is_prime_u64(x as u64)
                    } else {
                        false
                    }
                } else {
                    false
                }
            })),
        )
        & (w, PatternRestriction::Length(0, None));

    println!(
        "> Matching pattern {} : 0 <= len(x) <= 2, 0 <= len(y) <= 4, len(x) >= len(y) & is_prime(z) to {}:",
        pat_expr.printer(&state),
        expr.printer(&state)
    );

    let mut it = pattern.pattern_match(expr.as_view(), &state, &conditions);
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
