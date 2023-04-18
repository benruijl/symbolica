use ahash::HashMap;
use rug::Rational;
use symbolica::{
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
    rings::finite_field::{self, FiniteFieldU64, PrimeIteratorU64},
    state::{ResettableBuffer, State, Workspace},
};

fn expression_test() {
    let mut state = State::new();
    let workspace = Workspace::new();

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

    assert_eq!(a, b.to_tree());

    let view = b.to_view();

    println!(
        "> Raw input = {}, atom bytes = {}, rep bytes = {}",
        AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state),
        a.len(),
        b.len()
    );

    let mut normalized_handle = workspace.new_atom();

    let normalized = normalized_handle.get_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "\tout = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get().len());
}

fn finite_field_test() {
    let mut state = State::new();
    let workspace = Workspace::new();

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

    assert_eq!(a, b.to_tree(),);

    let view = b.to_view();

    println!(
        "> Raw input = {}, atom bytes = {}, rep bytes = {}",
        AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state),
        a.len(),
        b.len()
    );

    let mut normalized_handle = workspace.new_atom();

    let normalized = normalized_handle.get_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "\tout = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get().len());
}

fn parse_test() {
    let mut state = State::new();
    let workspace = Workspace::new();

    // spaces and underscores are allowed in numbers are are all stripped
    let token = parse("(1+  x^2/5  )*443_555*f(\t2*1,\n4* 44 5 + \r\n 2)^5\\*6").unwrap();

    let a = token.to_atom_tree(&mut state).unwrap();

    let mut b = OwnedAtom::new();
    b.from_tree(&a);

    assert_eq!(a, b.to_tree());

    let view = b.to_view();

    println!(
        "> Raw input = {}, atom bytes = {}, rep bytes = {}",
        AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state),
        a.len(),
        b.len()
    );

    let mut normalized_handle = workspace.new_atom();
    let normalized = normalized_handle.get_mut();

    view.normalize(&workspace, &state, normalized);

    print!(
        "\tout = {}",
        AtomPrinter::new(
            normalized.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
    );
    println!(", rep bytes = {}", normalized_handle.get().len());
}

fn pattern_test_1() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("x*y*w*z*f(x,y,x*y,z)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pat_expr = parse("z*x_*y_*g_(z_,x_,w_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pattern = Pattern::from_view(pat_expr.to_view(), &state);
    let mut restrictions = HashMap::default();
    restrictions.insert(
        state.get_or_insert_var("x_"),
        vec![PatternRestriction::Length(1, Some(100))],
    );

    println!(
        "> Matching pattern {} to {}:",
        AtomPrinter::new(
            pat_expr.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::Form, &state)
    );

    let mut it = PatternAtomTreeIterator::new(&pattern, expr.to_view(), &state, &restrictions);
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
        AtomPrinter::new(pat_expr.to_view(), symbolica::printer::PrintMode::Form, &state),
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::Form, &state)
    );

    let mut it = PatternAtomTreeIterator::new(&pattern, expr.to_view(), &state, &restrictions);
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
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(z)*f(f(x),z)*f(y)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    println!(
        "> Tree walk of {}:",
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::Form, &state)
    );

    for (loc, view) in AtomTreeIterator::new(expr.to_view()) {
        println!(
            "\tAtom at location {:?}: {}",
            loc,
            AtomPrinter::new(view, symbolica::printer::PrintMode::Form, &state)
        );
    }
}

fn tree_replace_test() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(z)*f(f(x))*f(y)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let pat_expr = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pattern = Pattern::from_view(pat_expr.to_view(), &state);
    let restrictions = HashMap::default();

    println!(
        "> Matching pattern {} to {}:",
        AtomPrinter::new(
            pat_expr.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::Form, &state)
    );

    let mut it = PatternAtomTreeIterator::new(&pattern, expr.to_view(), &state, &restrictions);
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
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(z)*f(f(x))*f(y)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let pat_expr = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let rhs_expr = parse("g(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let rhs = Pattern::from_view(rhs_expr.to_view(), &state);

    let pattern = Pattern::from_view(pat_expr.to_view(), &state);
    let restrictions = HashMap::default();

    println!(
        "> Replace once {}={} in {}:",
        AtomPrinter::new(
            pat_expr.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        AtomPrinter::new(
            rhs_expr.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::Form, &state),
    );

    let mut replaced = OwnedAtom::new();

    let mut it = ReplaceIterator::new(&pattern, expr.to_view(), &rhs, &state, &restrictions);
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
    let workspace = Workspace::new();

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(z)*f(f(x))*h(f(3))")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let pat_expr = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pattern = Pattern::from_view(pat_expr.to_view(), &state);
    let restrictions = HashMap::default();

    let rhs_expr = parse("g(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();
    let rhs = Pattern::from_view(rhs_expr.to_view(), &state);

    let mut out = OwnedAtom::new();

    pattern.replace_all(
        expr.to_view(),
        &rhs,
        &state,
        &workspace,
        &restrictions,
        &mut out,
    );

    println!(
        "> Replace all {}={} in {}: {}",
        AtomPrinter::new(
            pat_expr.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        AtomPrinter::new(
            rhs_expr.to_view(),
            symbolica::printer::PrintMode::Form,
            &state
        ),
        AtomPrinter::new(expr.to_view(), symbolica::printer::PrintMode::Form, &state),
        AtomPrinter::new(out.to_view(), symbolica::printer::PrintMode::Form, &state)
    );
}

fn fibonacci_test() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let pattern = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let mut restrictions = HashMap::default();
    restrictions.insert(
        state.get_or_insert_var("x_"),
        vec![PatternRestriction::Filter(Box::new(
            |v: &Match<DefaultRepresentation>| match v {
                Match::Single(v) => {
                    if let AtomView::Num(n) = v {
                        !n.is_one() && !n.is_zero()
                    } else {
                        false
                    }
                }
                _ => false,
            },
        ))],
    );

    let rhs = parse("f(x_ -1) + f(x_ - 2)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    // replace f(0) and f(1) by 1
    let lhs_zero_pat = parse("f(0)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let lhs_one_pat = parse("f(1)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let rhs_one = parse("1")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let expand_pat = parse("(x_+y_)*z_")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let expand_rhs = parse("x_*z_+y_*z_")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(10)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    for _ in 0..1 {
        let mut target = workspace.new_atom();
        target.get_mut().from_view(&expr.to_view());

        println!(
            "> Repeated calls of f(x_) = f(x_ - 1) + f(x_ - 2) on {}:",
            AtomPrinter::new(
                target.get().to_view(),
                symbolica::printer::PrintMode::Form,
                &state
            ),
        );

        for _ in 0..9 {
            let mut out = workspace.new_atom();
            pattern.replace_all(
                target.get().to_view(),
                &rhs,
                &state,
                &workspace,
                &restrictions,
                out.get_mut(),
            );

            // expand (f(1)+f(2))*4
            let mut out2 = workspace.new_atom();
            expand_pat.replace_all(
                out.get().to_view(),
                &expand_rhs,
                &state,
                &workspace,
                &HashMap::default(),
                out2.get_mut(),
            );

            // sort the expression
            let mut out_renom = workspace.new_atom();
            out2.get()
                .to_view()
                .normalize(&workspace, &state, out_renom.get_mut());
            out2 = out_renom;

            let mut out_renom2 = workspace.new_atom();
            lhs_zero_pat.replace_all(
                out2.get().to_view(),
                &rhs_one,
                &state,
                &workspace,
                &HashMap::default(),
                out_renom2.get_mut(),
            );

            let mut out3 = workspace.new_atom();
            lhs_one_pat.replace_all(
                out_renom2.get().to_view(),
                &rhs_one,
                &state,
                &workspace,
                &HashMap::default(),
                out3.get_mut(),
            );

            // sort expression
            let mut out_renom = workspace.new_atom();
            out3.get()
                .to_view()
                .normalize(&workspace, &state, out_renom.get_mut());

            println!(
                "\t{}",
                AtomPrinter::new(
                    out_renom.get().to_view(),
                    symbolica::printer::PrintMode::Form,
                    &state
                ),
            );

            target = out_renom;
        }
    }
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
    fibonacci_test();
}
