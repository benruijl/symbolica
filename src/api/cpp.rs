use std::ffi::{c_char, CStr};
use std::fmt::Write;
use std::os::raw::c_ulonglong;

use smartstring::{LazyCompact, SmartString};

use crate::parser::Token;
use crate::representations::Identifier;
use crate::rings::finite_field::{FiniteField, FiniteFieldCore};
use crate::rings::integer::IntegerRing;
use crate::rings::rational::RationalField;
use crate::{
    printer::{PrintOptions, RationalPolynomialPrinter},
    representations::default::Linear,
    rings::rational_polynomial::RationalPolynomial,
    state::{State, Workspace},
};

struct LocalState {
    buffer: String,
    var_map: Vec<Identifier>,
    var_name_map: Vec<SmartString<LazyCompact>>,
    input_has_rational_numbers: bool,
    exp_fits_in_u8: bool,
}

struct Symbolica {
    state: State,
    workspace: Workspace<Linear>,
    local_state: LocalState,
}

/// Create a new Symbolica handle.
#[no_mangle]
unsafe extern "C" fn init() -> *mut Symbolica {
    let s = Symbolica {
        state: State::new(),
        workspace: Workspace::new(),
        local_state: LocalState {
            buffer: String::with_capacity(2048),
            var_map: vec![],
            var_name_map: vec![],
            input_has_rational_numbers: false,
            exp_fits_in_u8: true,
        },
    };

    Box::into_raw(Box::new(s))
}

#[no_mangle]
unsafe extern "C" fn set_options(
    symbolica: *mut Symbolica,
    input_has_rational_numbers: bool,
    exp_fits_in_u8: bool,
) {
    let symbolica = unsafe { &mut *symbolica };

    symbolica.local_state.input_has_rational_numbers = input_has_rational_numbers;
    symbolica.local_state.exp_fits_in_u8 = exp_fits_in_u8;
}

#[no_mangle]
unsafe extern "C" fn set_vars(symbolica: *mut Symbolica, vars: *const c_char) {
    let c = unsafe { CStr::from_ptr(vars) };
    let cstr = c.to_str().unwrap();

    let symbolica = unsafe { &mut *symbolica };

    symbolica.local_state.var_map.clear();

    for var in cstr.split(',') {
        symbolica
            .local_state
            .var_map
            .push(symbolica.state.get_or_insert_var(var));
        symbolica.local_state.var_name_map.push(var.into());
    }
}

/// Simplify a rational polynomial. The return value is only valid until the next call to
/// `simplify`.
#[no_mangle]
unsafe extern "C" fn simplify(
    symbolica: *mut Symbolica,
    input: *const c_char,
    prime: c_ulonglong,
    explicit_rational_polynomial: bool,
) -> *const c_char {
    let c = unsafe { CStr::from_ptr(input) };
    let cstr = c.to_str().unwrap();

    let symbolica = unsafe { &mut *symbolica };

    let token = Token::parse(cstr).unwrap();

    macro_rules! to_rational {
        ($in_field: ty, $exp_size: ty) => {
            if prime == 0 {
                let r: RationalPolynomial<IntegerRing, $exp_size> = token
                    .to_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        <$in_field>::new(),
                        IntegerRing::new(),
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    RationalPolynomialPrinter {
                        poly: &r,
                        state: &symbolica.state,
                        opts: PrintOptions {
                            terms_on_new_line: false,
                            color_top_level_sum: false,
                            color_builtin_functions: false,
                            print_finite_field: false,
                            explicit_rational_polynomial
                        }
                    }
                )
                .unwrap();
            } else {
                if prime <= u32::MAX as c_ulonglong {
                    let field = FiniteField::<u32>::new(prime as u32);
                    let rf: RationalPolynomial<FiniteField<u32>, $exp_size> = token
                        .to_rational_polynomial(
                            &symbolica.workspace,
                            &mut symbolica.state,
                            field,
                            field,
                            &symbolica.local_state.var_map,
                            &symbolica.local_state.var_name_map,
                        )
                        .unwrap();

                    symbolica.local_state.buffer.clear();
                    write!(
                        &mut symbolica.local_state.buffer,
                        "{}\0", // add the NUL character
                        RationalPolynomialPrinter {
                            poly: &rf,
                            state: &symbolica.state,
                            opts: PrintOptions {
                                terms_on_new_line: false,
                                color_top_level_sum: false,
                                color_builtin_functions: false,
                                print_finite_field: false,
                                explicit_rational_polynomial
                            }
                        }
                    )
                    .unwrap();
                } else {
                    let field = FiniteField::<u64>::new(prime as u64);
                    let rf: RationalPolynomial<FiniteField<u64>, $exp_size> = token
                        .to_rational_polynomial(
                            &symbolica.workspace,
                            &mut symbolica.state,
                            field,
                            field,
                            &symbolica.local_state.var_map,
                            &symbolica.local_state.var_name_map,
                        )
                        .unwrap();

                    symbolica.local_state.buffer.clear();
                    write!(
                        &mut symbolica.local_state.buffer,
                        "{}\0", // add the NUL character
                        RationalPolynomialPrinter {
                            poly: &rf,
                            state: &symbolica.state,
                            opts: PrintOptions {
                                terms_on_new_line: false,
                                color_top_level_sum: false,
                                color_builtin_functions: false,
                                print_finite_field: false,
                                explicit_rational_polynomial
                            }
                        }
                    )
                    .unwrap();
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

    unsafe { CStr::from_bytes_with_nul_unchecked(symbolica.local_state.buffer.as_bytes()) }.as_ptr()
}

/// Free the Symbolica handle.
#[no_mangle]
unsafe extern "C" fn drop(symbolica: *mut Symbolica) {
    let _ = Box::from_raw(symbolica);
}
