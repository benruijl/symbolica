use wolfram_library_link::{self as wll};

use std::borrow::BorrowMut;
use std::sync::{Arc, RwLock};

use smartstring::{LazyCompact, SmartString};

use crate::domains::finite_field::{FiniteField, FiniteFieldCore};
use crate::domains::integer::IntegerRing;
use crate::domains::rational::RationalField;
use crate::parser::Token;
use crate::poly::Variable;
use crate::{
    domains::rational_polynomial::RationalPolynomial,
    printer::{PrintOptions, RationalPolynomialPrinter},
    state::{State, Workspace},
};
use once_cell::sync::Lazy;

static STATE: Lazy<RwLock<LocalState>> = Lazy::new(|| {
    RwLock::new(LocalState {
        buffer: String::with_capacity(2048),
        var_map: Arc::new(vec![]),
        var_name_map: vec![],
        input_has_rational_numbers: false,
        exp_fits_in_u8: true,
    })
});

struct LocalState {
    buffer: String,
    var_map: Arc<Vec<Variable>>,
    var_name_map: Vec<SmartString<LazyCompact>>,
    input_has_rational_numbers: bool,
    exp_fits_in_u8: bool,
}

#[wll::export(name = "SymbolicaSetOptions")]
fn set_options(input_has_rational_numbers: bool, exp_fits_in_u8: bool) {
    let mut symbolica = STATE.write().unwrap();
    symbolica.input_has_rational_numbers = input_has_rational_numbers;
    symbolica.exp_fits_in_u8 = exp_fits_in_u8;
}

#[wll::export(name = "SymbolicaSetVariables")]
fn set_vars(vars: String) {
    let mut symbolica = STATE.write().unwrap();

    let mut var_map = vec![];
    for var in vars.split(',') {
        let v = State::get_or_insert_var(var);
        var_map.push(v.into());
        symbolica.var_name_map.push(var.into());
    }

    symbolica.var_map = Arc::new(var_map);
}

/// Simplify a rational polynomial.
#[wll::export(name = "SymbolicaSimplify")]
fn simplify(input: String, prime: i64, explicit_rational_polynomial: bool) -> String {
    let mut symbolica = STATE.write().unwrap();
    let symbolica: &mut LocalState = symbolica.borrow_mut();

    let token = Token::parse(&input).unwrap();

    macro_rules! to_rational {
        ($in_field: ty, $exp_size: ty) => {
            if prime == 0 {
                let r: RationalPolynomial<IntegerRing, $exp_size> = Workspace::get_local()
                    .with(|workspace| {
                        token.to_rational_polynomial(
                            &workspace,
                            &<$in_field>::new(),
                            &Z,
                            &symbolica.var_map,
                            &symbolica.var_name_map,
                        )
                    })
                    .unwrap();

                format!(
                    "{}",
                    RationalPolynomialPrinter {
                        poly: &r,
                        opts: PrintOptions {
                            terms_on_new_line: false,
                            color_top_level_sum: false,
                            color_builtin_functions: false,
                            print_finite_field: false,
                            explicit_rational_polynomial,
                            symmetric_representation_for_finite_field: false,
                            number_thousands_separator: None,
                            multiplication_operator: '*',
                            square_brackets_for_function: false,
                            num_exp_as_superscript: false,
                            latex: false,
                        },
                        add_parentheses: false
                    }
                )
            } else {
                if prime >= 0 && prime <= u32::MAX as i64 {
                    let field = Zp::new(prime as u32);
                    let rf: RationalPolynomial<Zp, $exp_size> = Workspace::get_local()
                        .with(|workspace| {
                            token.to_rational_polynomial(
                                &workspace,
                                &field,
                                &field,
                                &symbolica.var_map,
                                &symbolica.var_name_map,
                            )
                        })
                        .unwrap();

                    symbolica.buffer.clear();
                    format!(
                        "{}",
                        RationalPolynomialPrinter {
                            poly: &rf,
                            opts: PrintOptions {
                                terms_on_new_line: false,
                                color_top_level_sum: false,
                                color_builtin_functions: false,
                                print_finite_field: false,
                                explicit_rational_polynomial,
                                symmetric_representation_for_finite_field: false,
                                number_thousands_separator: None,
                                multiplication_operator: '*',
                                square_brackets_for_function: false,
                                num_exp_as_superscript: false,
                                latex: false,
                            },
                            add_parentheses: false
                        }
                    )
                } else {
                    let field = Zp64::new(prime as u64);
                    let rf: RationalPolynomial<Zp64, $exp_size> = Workspace::get_local()
                        .with(|workspace| {
                            token.to_rational_polynomial(
                                &workspace,
                                &field,
                                &field,
                                &symbolica.var_map,
                                &symbolica.var_name_map,
                            )
                        })
                        .unwrap();

                    symbolica.buffer.clear();
                    format!(
                        "{}",
                        RationalPolynomialPrinter {
                            poly: &rf,
                            opts: PrintOptions {
                                terms_on_new_line: false,
                                color_top_level_sum: false,
                                color_builtin_functions: false,
                                print_finite_field: false,
                                explicit_rational_polynomial,
                                symmetric_representation_for_finite_field: false,
                                number_thousands_separator: None,
                                multiplication_operator: '*',
                                square_brackets_for_function: false,
                                num_exp_as_superscript: false,
                                latex: false,
                            },
                            add_parentheses: false
                        }
                    )
                }
            }
        };
    }

    match (
        symbolica.input_has_rational_numbers,
        symbolica.exp_fits_in_u8,
    ) {
        (false, true) => to_rational!(IntegerRing, u8),
        (true, true) => to_rational!(RationalField, u8),
        (false, false) => to_rational!(IntegerRing, u16),
        (true, false) => to_rational!(RationalField, u16),
    }
}
