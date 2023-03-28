use ahash::HashMap;
use rug::Rational;
use symbolica::{
    finite_field::{self, FiniteFieldU64, PrimeIteratorU64},
    id::{
        AtomTreeIterator, Match, Pattern, PatternAtomTreeIterator, PatternRestriction,
        ReplaceIterator,
    },
    parser::parse,
    printer::AtomPrinter,
    representations::{
        default::DefaultRepresentation,
        number::{BorrowedNumber, Number},
        tree::AtomTree,
        AtomView, Num, OwnedAtom,
    },
    state::{ResettableBuffer, State, Workspace},
};

fn expression_test() {
    let mut state = State::new();

    // create variable names
    let (x, y, z) = (
        state.get_or_insert_var("x"),
        state.get_or_insert_var("y"),
        state.get_or_insert_var("z"),
    );

    // create term
    let a = AtomTree::Mul(vec![
        AtomTree::Pow(Box::new((
            AtomTree::Var(x),
            AtomTree::Num(Number::Natural(2, 1)),
        ))),
        AtomTree::Num(Number::Natural(3, 1)),
        AtomTree::Pow(Box::new((
            AtomTree::Var(x),
            AtomTree::Num(Number::Natural(-2, 1)),
        ))),
        AtomTree::Var(y),
        AtomTree::Num(Number::Large(
            Rational::from_str_radix(
                "1723671261273182378912738921/128937127893761293712893712983712",
                10,
            )
            .unwrap(),
        )),
        AtomTree::Var(y),
        AtomTree::Fn(
            z,
            vec![AtomTree::Add(vec![
                AtomTree::Num(Number::Natural(1, 1)),
                AtomTree::Var(x),
            ])],
        ),
    ]);

    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    assert!(
        a == b.to_tree(),
        "Not similar: {:?} vs {:?}",
        a,
        b.to_tree()
    );

    let view = b.to_view();

    println!(
        "> Raw input = {}, atom bytes = {}, rep bytes = {}",
        AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state),
        a.len(),
        b.len()
    );

    let workspace = Workspace::new();

    let mut normalized_handle = workspace.new_atom();

    let normalized = normalized_handle.get_buf_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "\tout = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get_buf().len());
}

fn finite_field_test() {
    let mut state = State::new();

    let prime = PrimeIteratorU64::new(16).next().unwrap();
    let f = FiniteFieldU64::new(prime);
    let fi = state.get_or_insert_finite_field(f.clone());

    let x = state.get_or_insert_var("x");

    // create term x * (13 % 17) * (15 % 17)
    let a = AtomTree::Mul(vec![
        AtomTree::Num(Number::FiniteField(f.to_montgomery(13), fi)),
        AtomTree::Num(Number::FiniteField(f.to_montgomery(15), fi)),
        AtomTree::Var(x),
    ]);

    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    assert!(
        a == b.to_tree(),
        "Not similar: {:?} vs {:?}",
        a,
        b.to_tree()
    );

    let view = b.to_view();

    println!(
        "> Raw input = {}, atom bytes = {}, rep bytes = {}",
        AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state),
        a.len(),
        b.len()
    );

    let workspace = Workspace::new();

    let mut normalized_handle = workspace.new_atom();

    let normalized = normalized_handle.get_buf_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "\tout = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get_buf().len());
}

fn parse_test() {
    let mut state = State::new();

    // spaces and underscores are allowed in numbers are are all stripped
    let token = parse("(1+  x^2/5  )*443_555*f(\t2*1,\n4* 44 5 + \r\n 2)^5\\*6").unwrap();

    let a = token.to_atom_tree(&mut state).unwrap();

    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    assert!(
        a == b.to_tree(),
        "Not similar: {:?} vs {:?}",
        a,
        b.to_tree()
    );

    let view = b.to_view();

    println!(
        "> Raw input = {}, atom bytes = {}, rep bytes = {}",
        AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state),
        a.len(),
        b.len()
    );

    let workspace = Workspace::new();

    let mut normalized_handle = workspace.new_atom();
    let normalized = normalized_handle.get_buf_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "\tout = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get_buf().len());
}

fn pattern_test_1() {
    let mut state = State::new();

    let token = parse("x*y*w*z*f(x,y,x*y,z)").unwrap();
    let a = token.to_atom_tree(&mut state).unwrap();
    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    let pattern_token = parse("z*x_*y_*g_(z_,x_,w_)").unwrap();
    let pat_a = pattern_token.to_atom_tree(&mut state).unwrap();
    let mut pat_b = OwnedAtom::new();
    pat_b.from_tree(&pat_a);

    let pattern = Pattern::from_view(pat_b.to_view(), &state);
    let mut restrictions = HashMap::default();
    restrictions.insert(
        state.get_or_insert_var("x_"),
        vec![PatternRestriction::Length(1, Some(100))],
        /*  vec![PatternRestriction::Filter(Box::new(|x| match x {
            Match::Single(_) => true,
            _ => false,
        }))],*/
    );

    let mut it = PatternAtomTreeIterator::new(&pattern, b.to_view(), &state, &restrictions);

    println!(
        "> Matching pattern {} to {}:",
        AtomPrinter::new(pat_b.to_view(), symbolica::printer::PrintMode::Form, &state),
        AtomPrinter::new(b.to_view(), symbolica::printer::PrintMode::Form, &state)
    );

    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\t Match at location {:?} - {:?}:", location, used_flags);
        for (id, v) in match_stack {
            print!("\t\t{} = ", state.get_name(*id).unwrap());
            match v {
                Match::Single(s) => print!(
                    "{}",
                    AtomPrinter::new(*s, symbolica::printer::PrintMode::Form, &state),
                ),
                Match::Multiple(slice_type, mm) => {
                    print!("{:?} ", slice_type);
                    for vv in mm {
                        print!(
                            "{}",
                            AtomPrinter::new(*vv, symbolica::printer::PrintMode::Form, &state),
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

fn pattern_test_2() {
    let mut state = State::new();

    let token = parse("f(1,2,3,4,5,6,7)").unwrap();
    let a = token.to_atom_tree(&mut state).unwrap();
    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    let pattern_token = parse("f(x_,y_,z_,w_)").unwrap();
    let pat_a = pattern_token.to_atom_tree(&mut state).unwrap();
    let mut pat_b = OwnedAtom::new();
    pat_b.from_tree(&pat_a);

    let pattern = Pattern::from_view(pat_b.to_view(), &state);
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
                    let len_x = if let Match::Multiple(_, s) = x {
                        s.len()
                    } else {
                        1
                    };
                    let len_y = if let Match::Multiple(_, s) = y {
                        s.len()
                    } else {
                        1
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

    let mut it = PatternAtomTreeIterator::new(&pattern, b.to_view(), &state, &restrictions);

    println!(
        "> Matching pattern {} : 0 <= len(x) <= 2, 0 <= len(y) <= 4, len(x) >= len(y) & is_prime(z) to {}:",
        AtomPrinter::new(pat_b.to_view(), symbolica::printer::PrintMode::Form, &state),
        AtomPrinter::new(b.to_view(), symbolica::printer::PrintMode::Form, &state)
    );

    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\tMatch at location {:?} - {:?}:", location, used_flags);
        for (id, v) in match_stack {
            print!("\t\t{} = ", state.get_name(*id).unwrap());
            match v {
                Match::Single(s) => print!(
                    "{}",
                    AtomPrinter::new(*s, symbolica::printer::PrintMode::Form, &state),
                ),
                Match::Multiple(slice_type, mm) => {
                    print!("{:?} ", slice_type);
                    for vv in mm {
                        print!(
                            "{}",
                            AtomPrinter::new(*vv, symbolica::printer::PrintMode::Form, &state),
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

fn tree_walk_test() {
    let mut state = State::new();

    let token = parse("f(z)*f(f(x),z)*f(y)").unwrap();
    let a = token.to_atom_tree(&mut state).unwrap();
    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    println!(
        "> Tree walk of {}:",
        AtomPrinter::new(b.to_view(), symbolica::printer::PrintMode::Form, &state)
    );

    let mut it = AtomTreeIterator::new(b.to_view());
    while let Some((loc, view)) = it.next() {
        println!(
            "\tAtom at location {:?}: {}",
            loc,
            AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state)
        );
    }
}

fn tree_replace_test() {
    let mut state = State::new();

    let token = parse("f(z)*f(f(x))*f(y)").unwrap();
    let a = token.to_atom_tree(&mut state).unwrap();
    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    let pattern_token = parse("f(x_)").unwrap();
    let pat_a = pattern_token.to_atom_tree(&mut state).unwrap();
    let mut pat_b = OwnedAtom::new();
    pat_b.from_tree(&pat_a);

    let pattern = Pattern::from_view(pat_b.to_view(), &state);
    let restrictions = HashMap::default();

    let mut it = PatternAtomTreeIterator::new(&pattern, b.to_view(), &state, &restrictions);

    println!(
        "> Matching pattern {} to {}:",
        AtomPrinter::new(pat_b.to_view(), symbolica::printer::PrintMode::Form, &state),
        AtomPrinter::new(b.to_view(), symbolica::printer::PrintMode::Form, &state)
    );

    while let Some((location, used_flags, _atom, match_stack)) = it.next() {
        println!("\tMatch at location {:?} - {:?}:", location, used_flags);
        for (id, v) in match_stack {
            print!("\t\t{} = ", state.get_name(*id).unwrap());
            match v {
                Match::Single(s) => print!(
                    "{}",
                    AtomPrinter::new(*s, symbolica::printer::PrintMode::Form, &state),
                ),
                Match::Multiple(slice_type, mm) => {
                    print!("{:?} ", slice_type);
                    for vv in mm {
                        print!(
                            "{}",
                            AtomPrinter::new(*vv, symbolica::printer::PrintMode::Form, &state),
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

fn replace_once_test() {
    let mut state = State::new();

    let token = parse("f(z)*f(f(x))*f(y)").unwrap();
    let a = token.to_atom_tree(&mut state).unwrap();
    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    let pattern_token = parse("f(x_)").unwrap();
    let pat_a = pattern_token.to_atom_tree(&mut state).unwrap();
    let mut pat_b = OwnedAtom::new();
    pat_b.from_tree(&pat_a);

    let rhs_token = parse("g(x_)").unwrap();
    let pat_rhs = rhs_token.to_atom_tree(&mut state).unwrap();
    let mut at_rhs = OwnedAtom::new();
    at_rhs.from_tree(&pat_rhs);
    let rhs = Pattern::from_view(at_rhs.to_view(), &state);

    let pattern = Pattern::from_view(pat_b.to_view(), &state);
    let restrictions = HashMap::default();

    let mut it = ReplaceIterator::new(&pattern, b.to_view(), &rhs, &state, &restrictions);

    println!(
        "> Replace once {}={} in {}:",
        AtomPrinter::new(pat_b.to_view(), symbolica::printer::PrintMode::Form, &state),
        AtomPrinter::new(
            at_rhs.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        AtomPrinter::new(b.to_view(), symbolica::printer::PrintMode::Form, &state),
    );

    let workspace = Workspace::new();
    let mut replaced = OwnedAtom::new();

    while let Some(()) = it.next(&workspace, &mut replaced) {
        println!(
            "\t{}",
            AtomPrinter::new(
                replaced.to_view(),
                symbolica::printer::PrintMode::Form,
                &state
            ),
        );
    }
}

fn replace_all_test() {
    let mut state = State::new();

    let token = parse("f(z)*f(f(x))*h(f(3))").unwrap();
    let a = token.to_atom_tree(&mut state).unwrap();
    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    let pattern_token = parse("f(x_)").unwrap();
    let pat_a = pattern_token.to_atom_tree(&mut state).unwrap();
    let mut pat_b = OwnedAtom::new();
    pat_b.from_tree(&pat_a);

    let pattern = Pattern::from_view(pat_b.to_view(), &state);
    let restrictions = HashMap::default();

    let rhs_token = parse("g(x_)").unwrap();
    let pat_rhs = rhs_token.to_atom_tree(&mut state).unwrap();
    let mut at_rhs = OwnedAtom::new();
    at_rhs.from_tree(&pat_rhs);
    let rhs = Pattern::from_view(at_rhs.to_view(), &state);

    let workspace = Workspace::new();

    let mut out = OwnedAtom::new();

    pattern.replace_all(
        b.to_view(),
        &rhs,
        &state,
        &workspace,
        &restrictions,
        &mut out,
    );

    println!(
        "> Replace all {}={} in {}: {}",
        AtomPrinter::new(pat_b.to_view(), symbolica::printer::PrintMode::Form, &state),
        AtomPrinter::new(
            at_rhs.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        AtomPrinter::new(b.to_view(), symbolica::printer::PrintMode::Form, &state),
        AtomPrinter::new(out.to_view(), symbolica::printer::PrintMode::Form, &state)
    );
}

fn main() {
    expression_test();
    finite_field_test();
    parse_test();
    pattern_test_1();
    pattern_test_2();
    tree_walk_test();
    tree_replace_test();
    replace_once_test();
    replace_all_test();
}
