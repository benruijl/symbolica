use ahash::HashMap;
use symbolica::{
    id::{Match, Pattern, PatternAtomTreeIterator, PatternRestriction},
    parser::parse,
    printer::AtomPrinter,
    representations::{default::DefaultRepresentation, OwnedAtom, AtomView, number::BorrowedNumber, Num},
    state::{State, Workspace}, rings::finite_field,
};
fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(1,2,3,4,5,6,7)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pat_expr = parse("f(x_,y_,z_,w_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pattern = Pattern::from_view(pat_expr.to_view(), &state);
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
        vec![PatternRestriction::Filter(Box::new(
            |x: &Match<DefaultRepresentation>| {
                if let Match::Single(s) = x {
                    if let AtomView::Num(num) = s {
                        if let BorrowedNumber::Natural(x, y) = num.get_number_view() {
                            y == 1 && x > 0 && finite_field::is_prime_u64(x as u64)
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            },
        ))],
    );
    restrictions.insert(
        state.get_or_insert_var("w_"),
        vec![PatternRestriction::Length(0, None)],
    );

    println!(
        "> Matching pattern {} : 0 <= len(x) <= 2, 0 <= len(y) <= 4, len(x) >= len(y) & is_prime(z) to {}:",
        AtomPrinter::new(pat_expr.to_view(), symbolica::printer::PrintMode::default(), &state),
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::default(), &state)
    );

    let mut it = PatternAtomTreeIterator::new(&pattern, expr.to_view(), &state, &restrictions);
    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\tMatch at location {:?} - {:?}:", location, used_flags);
        for (id, v) in match_stack {
            print!("\t\t{} = ", state.get_name(*id).unwrap());
            match v {
                Match::Single(s) => print!(
                    "{}",
                    AtomPrinter::new(*s, symbolica::printer::PrintMode::default(), &state),
                ),
                Match::Multiple(slice_type, mm) => {
                    print!("{:?} ", slice_type);
                    for vv in mm {
                        print!(
                            "{}",
                            AtomPrinter::new(*vv, symbolica::printer::PrintMode::default(), &state),
                        );
                        print!(", ")
                    }
                }
                Match::FunctionName(f) => {
                    print!("Fn {}", state.get_name(*f).unwrap())
                }
            }
            println!("");
        }
    }
}