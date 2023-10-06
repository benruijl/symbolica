use wolfram_library_link::{self as wll};

use std::borrow::BorrowMut;
use std::sync::RwLock;

use smartstring::{LazyCompact, SmartString};

use crate::parser::Token;
use crate::poly::Variable;
use crate::rings::finite_field::{FiniteField, FiniteFieldCore};
use crate::rings::integer::IntegerRing;
use crate::rings::rational::RationalField;
use crate::{
    printer::{PrintOptions, RationalPolynomialPrinter},
    rings::rational_polynomial::RationalPolynomial,
    state::{State, Workspace},
};
use once_cell::sync::Lazy;

static STATE: Lazy<RwLock<Symbolica>> = Lazy::new(|| {
    RwLock::new(Symbolica {
        state: State::new(),
        local_state: LocalState {
            buffer: String::with_capacity(2048),
            var_map: vec![],
            var_name_map: vec![],
            input_has_rational_numbers: false,
            exp_fits_in_u8: true,
        },
    })
});
thread_local!(static WORKSPACE: Workspace = Workspace::new());

struct LocalState {
    buffer: String,
    var_map: Vec<Variable>,
    var_name_map: Vec<SmartString<LazyCompact>>,
    input_has_rational_numbers: bool,
    exp_fits_in_u8: bool,
}

struct Symbolica {
    state: State,
    local_state: LocalState,
}

#[wll::export(name = "SymbolicaSetOptions")]
fn set_options(input_has_rational_numbers: bool, exp_fits_in_u8: bool) {
    let mut symbolica = STATE.write().unwrap();
    symbolica.local_state.input_has_rational_numbers = input_has_rational_numbers;
    symbolica.local_state.exp_fits_in_u8 = exp_fits_in_u8;
}

#[wll::export(name = "SymbolicaSetVariables")]
fn set_vars(vars: String) {
    let mut symbolica = STATE.write().unwrap();
    symbolica.local_state.var_map.clear();

    for var in vars.split(',') {
        let v = symbolica.state.get_or_insert_var(var);
        symbolica.local_state.var_map.push(v.into());
        symbolica.local_state.var_name_map.push(var.into());
    }
}

/// Simplify a rational polynomial.
#[wll::export(name = "SymbolicaSimplify")]
fn simplify(input: String, prime: i64, explicit_rational_polynomial: bool) -> String {
    let mut symbolica = STATE.write().unwrap();
    let symbolica: &mut Symbolica = symbolica.borrow_mut();

    let token = Token::parse(&input).unwrap();

    macro_rules! to_rational {
        ($in_field: ty, $exp_size: ty) => {
            if prime == 0 {
                let r: RationalPolynomial<IntegerRing, $exp_size> = WORKSPACE
                    .with(|workspace| {
                        token.to_rational_polynomial(
                            &workspace,
                            &mut symbolica.state,
                            <$in_field>::new(),
                            IntegerRing::new(),
                            &symbolica.local_state.var_map,
                            &symbolica.local_state.var_name_map,
                        )
                    })
                    .unwrap();

                format!(
                    "{}",
                    RationalPolynomialPrinter {
                        poly: &r,
                        state: &symbolica.state,
                        opts: PrintOptions {
                            terms_on_new_line: false,
                            color_top_level_sum: false,
                            color_builtin_functions: false,
                            print_finite_field: false,
                            explicit_rational_polynomial,
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
                    let field = FiniteField::<u32>::new(prime as u32);
                    let rf: RationalPolynomial<FiniteField<u32>, $exp_size> = WORKSPACE
                        .with(|workspace| {
                            token.to_rational_polynomial(
                                &workspace,
                                &mut symbolica.state,
                                field,
                                field,
                                &symbolica.local_state.var_map,
                                &symbolica.local_state.var_name_map,
                            )
                        })
                        .unwrap();

                    symbolica.local_state.buffer.clear();
                    format!(
                        "{}",
                        RationalPolynomialPrinter {
                            poly: &rf,
                            state: &symbolica.state,
                            opts: PrintOptions {
                                terms_on_new_line: false,
                                color_top_level_sum: false,
                                color_builtin_functions: false,
                                print_finite_field: false,
                                explicit_rational_polynomial,
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
                    let field = FiniteField::<u64>::new(prime as u64);
                    let rf: RationalPolynomial<FiniteField<u64>, $exp_size> = WORKSPACE
                        .with(|workspace| {
                            token.to_rational_polynomial(
                                &workspace,
                                &mut symbolica.state,
                                field,
                                field,
                                &symbolica.local_state.var_map,
                                &symbolica.local_state.var_name_map,
                            )
                        })
                        .unwrap();

                    symbolica.local_state.buffer.clear();
                    format!(
                        "{}",
                        RationalPolynomialPrinter {
                            poly: &rf,
                            state: &symbolica.state,
                            opts: PrintOptions {
                                terms_on_new_line: false,
                                color_top_level_sum: false,
                                color_builtin_functions: false,
                                print_finite_field: false,
                                explicit_rational_polynomial,
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
        symbolica.local_state.input_has_rational_numbers,
        symbolica.local_state.exp_fits_in_u8,
    ) {
        (false, true) => to_rational!(IntegerRing, u8),
        (true, true) => to_rational!(RationalField, u8),
        (false, false) => to_rational!(IntegerRing, u16),
        (true, false) => to_rational!(RationalField, u16),
    }
}
