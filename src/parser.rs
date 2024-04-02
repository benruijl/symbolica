use std::{fmt::Write, string::String, sync::Arc};

use bytes::Buf;
use rug::Integer as MultiPrecisionInteger;

use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

use crate::{
    coefficient::ConvertToRing,
    domains::{integer::Integer, Ring},
    poly::{polynomial::MultivariatePolynomial, Exponent, Variable},
    representations::Atom,
    state::{State, Workspace},
};

const HEX_DIGIT_MASK: [bool; 255] = [
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, true, true, true, true, true,
    true, true, true, true, true, false, false, false, false, false, false, false, true, true,
    true, true, true, true, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false,
];

const DIGIT_MASK: [bool; 255] = [
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, true, true, true, true, true,
    true, true, true, true, true, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false,
];

const HEX_TO_DIGIT: [u8; 24] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 0,
];

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ParseState {
    Identifier,
    Number,
    RationalPolynomial,
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operator {
    Mul,
    Add,
    Pow,
    Argument, // comma
    Neg,      // left side should be tagged as 'finished'
    Inv,      // left side should be tagged as 'finished', for internal use
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Mul => f.write_char('*'),
            Operator::Add => f.write_char('+'),
            Operator::Pow => f.write_char('^'),
            Operator::Argument => f.write_char(','),
            Operator::Neg => f.write_char('-'),
            Operator::Inv => f.write_char('/'),
        }
    }
}

impl Operator {
    #[inline]
    pub fn get_arity(&self) -> usize {
        match self {
            Operator::Neg | Operator::Inv => 1,
            _ => 2,
        }
    }

    #[inline]
    pub fn get_precedence(&self) -> u8 {
        match self {
            Operator::Mul => 8,
            Operator::Add => 7,
            Operator::Pow => 11,
            Operator::Argument => 6,
            Operator::Neg => 10,
            Operator::Inv => 9,
        }
    }

    #[inline]
    pub fn right_associative(&self) -> bool {
        match self {
            Operator::Mul => true,
            Operator::Add => true,
            Operator::Pow => false,
            Operator::Argument => true,
            Operator::Neg => true,
            Operator::Inv => true,
        }
    }
}

pub struct Position {
    pub line_number: usize,
    pub char_pos: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    Number(SmartString<LazyCompact>),
    ID(SmartString<LazyCompact>),
    RationalPolynomial(SmartString<LazyCompact>),
    Op(bool, bool, Operator, Vec<Token>),
    Fn(bool, Vec<Token>),
    Start,
    OpenParenthesis,
    CloseParenthesis,
    EOF,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Number(n) => f.write_str(n),
            Token::ID(v) => f.write_str(v),
            Token::RationalPolynomial(v) => {
                f.write_char('[')?;
                f.write_str(v)?;
                f.write_char(']')
            }
            Token::Op(_, _, o, m) => {
                let mut first = true;
                f.write_char('(')?;

                for mm in m {
                    if !first {
                        match o {
                            Operator::Mul => f.write_char('*')?,
                            Operator::Add => f.write_char('+')?,
                            Operator::Pow => f.write_char('^')?,
                            Operator::Argument => f.write_char(',')?,
                            Operator::Neg => f.write_char('-')?,
                            Operator::Inv => f.write_str("1/")?,
                        }
                    } else if *o == Operator::Neg {
                        f.write_char('-')?;
                    } else if *o == Operator::Inv {
                        f.write_str("1/")?;
                    }
                    first = false;

                    mm.fmt(f)?;
                }
                f.write_char(')')
            }
            Token::Fn(_, args) => {
                let mut first = true;

                match &args[0] {
                    Token::ID(s) => f.write_str(s)?,
                    _ => unreachable!(),
                };

                f.write_char('(')?;
                for aa in args.iter().skip(1) {
                    if !first {
                        f.write_char(',')?;
                    }
                    first = false;

                    aa.fmt(f)?;
                }
                f.write_char(')')
            }
            Token::Start => f.write_str("START"),
            Token::OpenParenthesis => f.write_char('('),
            Token::CloseParenthesis => f.write_char(')'),
            Token::EOF => f.write_str("EOF"),
        }
    }
}

impl Token {
    /// Return if the token does not require any further arguments.
    fn is_normal(&self) -> bool {
        match self {
            Token::Number(_) => true,
            Token::ID(_) => true,
            Token::RationalPolynomial(_) => true,
            Token::Op(more_left, more_right, _, _) => !more_left && !more_right,
            Token::Fn(more_right, _) => !more_right,
            _ => false,
        }
    }

    /// Get the precedence of the token.
    #[inline]
    fn get_precedence(&self) -> u8 {
        match self {
            Token::Number(_) => 11,
            Token::ID(_) => 11,
            Token::RationalPolynomial(_) => 11,
            Token::Op(_, _, o, _) => o.get_precedence(),
            Token::Fn(_, _) | Token::OpenParenthesis | Token::CloseParenthesis => 5,
            Token::Start | Token::EOF => 4,
        }
    }

    /// Add `other` to the left side of `self`, where `self` is a binary operation.
    #[inline]
    fn add_left(&mut self, other: Token) -> Result<(), String> {
        if let Token::Op(ml, _, o1, args) = self {
            debug_assert!(*ml);
            *ml = false;

            if let Token::Op(ml, mr, o2, mut args2) = other {
                debug_assert!(!ml && !mr);
                if *o1 == o2 {
                    // add from the left by swapping and then extending from the right
                    std::mem::swap(args, &mut args2);
                    args.append(&mut args2);
                } else {
                    args.insert(0, Token::Op(false, false, o2, args2));
                }
            } else {
                args.insert(0, other);
            }
            Ok(())
        } else {
            Err(format!(
                "operator expected, but found '{}'. Are parentheses unbalanced?",
                self
            ))
        }
    }

    fn distribute_neg(&mut self) {
        match self {
            Token::Op(_, _, Operator::Neg, args3) => {
                debug_assert!(args3.len() == 1);
                *self = args3.pop().unwrap();
            }
            Token::Op(_, _, Operator::Mul, args2) => {
                args2[0].distribute_neg();
            }
            Token::Op(_, _, Operator::Add, args2) => {
                for a in args2 {
                    a.distribute_neg();
                }
            }
            Token::Number(n) => {
                if n.starts_with('-') {
                    n.remove(0);
                } else {
                    n.insert(0, '-');
                }
            }
            _ => {
                let t = std::mem::replace(self, Token::EOF);
                *self = Token::Op(false, false, Operator::Neg, vec![t]);
            }
        }
    }

    /// Add `other` to right side of `self`, where `self` is a binary operation.
    #[inline]
    fn add_right(&mut self, mut other: Token) -> Result<(), String> {
        if let Token::Op(_, mr, o1, args) = self {
            debug_assert!(*mr);
            *mr = false;

            if *o1 == Operator::Neg {
                if let Token::Number(n) = &mut other {
                    if n.starts_with('-') {
                        n.remove(0);
                    } else {
                        n.insert(0, '-');
                    }
                } else {
                    other.distribute_neg();
                }
                *self = other;
                return Ok(());
            }

            if let Token::Op(ml, mr, o2, mut args2) = other {
                debug_assert!(!ml && !mr);
                if *o1 == o2 && o2.right_associative() {
                    if o2 == Operator::Neg || o2 == Operator::Inv {
                        // twice unary minus or inv cancels out
                        debug_assert!(args2.len() == 1);
                        *self = args2.pop().unwrap();
                    } else {
                        args.append(&mut args2)
                    }
                } else {
                    args.push(Token::Op(false, false, o2, args2));
                }
            } else {
                args.push(other);
            }

            Ok(())
        } else {
            Err(format!(
                "operator expected, but found '{}'. Are parentheses unbalanced?",
                self
            ))
        }
    }

    /// Parse the token into an atom.
    pub fn to_atom(&self, workspace: &Workspace) -> Result<Atom, String> {
        let mut atom = Atom::default();

        let mut state = State::get_global_state().write().unwrap();
        self.to_atom_with_output(&mut state, workspace, &mut atom)?;

        Ok(atom)
    }

    /// Parse the token into the atom `out`.
    fn to_atom_with_output(
        &self,
        state: &mut State,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> Result<(), String> {
        match self {
            Token::Number(n) => match n.parse::<Integer>() {
                Ok(x) => {
                    out.to_num(x.into());
                }
                Err(e) => return Err(format!("Could not parse number: {}", e)),
            },
            Token::ID(x) => {
                out.to_var(state.get_symbol_impl(x));
            }
            Token::Op(_, _, op, args) => match op {
                Operator::Mul => {
                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();

                    let mut atom = workspace.new_atom();
                    for a in args {
                        a.to_atom_with_output(state, workspace, &mut atom)?;
                        mul.extend(atom.as_view());
                    }

                    mul_h.as_view().normalize(workspace, out);
                }
                Operator::Add => {
                    let mut add_h = workspace.new_atom();
                    let add = add_h.to_add();

                    let mut atom = workspace.new_atom();
                    for a in args {
                        a.to_atom_with_output(state, workspace, &mut atom)?;
                        add.extend(atom.as_view());
                    }

                    add_h.as_view().normalize(workspace, out);
                }
                Operator::Pow => {
                    // pow is right associative
                    args.last()
                        .unwrap()
                        .to_atom_with_output(state, workspace, out)?;
                    for a in args.iter().rev().skip(1) {
                        let mut cur_base = workspace.new_atom();
                        a.to_atom_with_output(state, workspace, &mut cur_base)?;

                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(cur_base.as_view(), out.as_view());
                        pow_h.as_view().normalize(workspace, out);
                    }
                }
                Operator::Argument => return Err("Unexpected argument operator".into()),
                Operator::Neg => {
                    debug_assert!(args.len() == 1);

                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output(state, workspace, &mut base)?;

                    let num = workspace.new_num(-1);

                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();
                    mul.extend(base.as_view());
                    mul.extend(num.as_view());
                    mul_h.as_view().normalize(workspace, out);
                }
                Operator::Inv => {
                    debug_assert!(args.len() == 1);

                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output(state, workspace, &mut base)?;

                    let num = workspace.new_num(-1);

                    let mut pow_h = workspace.new_atom();
                    pow_h.to_pow(base.as_view(), num.as_view());
                    pow_h.as_view().normalize(workspace, out);
                }
            },
            Token::Fn(_, args) => {
                let name = match &args[0] {
                    Token::ID(s) => s,
                    _ => unreachable!(),
                };

                let mut fun_h = workspace.new_atom();
                let fun = fun_h.to_fun(state.get_symbol_impl(name));
                let mut atom = workspace.new_atom();
                for a in args.iter().skip(1) {
                    a.to_atom_with_output(state, workspace, &mut atom)?;
                    fun.add_arg(atom.as_view());
                }

                fun_h.as_view().normalize(workspace, out);
            }
            x => return Err(format!("Unexpected token {}", x)),
        }

        Ok(())
    }

    /// Parse the token into the atom `out` with pre-defined variables
    pub fn to_atom_with_output_and_var_map(
        &self,
        workspace: &Workspace,
        var_map: &Arc<Vec<Variable>>,
        var_name_map: &[SmartString<LazyCompact>],
        out: &mut Atom,
    ) -> Result<(), String> {
        match self {
            Token::Number(n) => {
                out.to_num(n.parse::<Integer>()?.into());
            }
            Token::ID(name) => {
                let index = var_name_map
                    .iter()
                    .position(|x| x == name)
                    .ok_or_else(|| format!("Undefined variable {}", name))?;
                if let Variable::Symbol(id) = var_map[index] {
                    out.to_var(id);
                } else {
                    Err(format!("Undefined variable {}", name))?;
                }
            }
            Token::Op(_, _, op, args) => match op {
                Operator::Mul => {
                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();

                    let mut atom = workspace.new_atom();
                    for a in args {
                        a.to_atom_with_output_and_var_map(
                            workspace,
                            var_map,
                            var_name_map,
                            &mut atom,
                        )?;
                        mul.extend(atom.as_view());
                    }

                    mul_h.as_view().normalize(workspace, out);
                }
                Operator::Add => {
                    let mut add_h = workspace.new_atom();
                    let add = add_h.to_add();

                    let mut atom = workspace.new_atom();
                    for a in args {
                        a.to_atom_with_output_and_var_map(
                            workspace,
                            var_map,
                            var_name_map,
                            &mut atom,
                        )?;
                        add.extend(atom.as_view());
                    }

                    add_h.as_view().normalize(workspace, out);
                }
                Operator::Pow => {
                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output_and_var_map(
                        workspace,
                        var_map,
                        var_name_map,
                        &mut base,
                    )?;

                    let mut exp = workspace.new_atom();
                    args[1].to_atom_with_output_and_var_map(
                        workspace,
                        var_map,
                        var_name_map,
                        &mut exp,
                    )?;

                    let mut pow_h = workspace.new_atom();
                    pow_h.to_pow(base.as_view(), exp.as_view());
                    pow_h.as_view().normalize(workspace, out);
                }
                Operator::Argument => return Err("Unexpected argument operator".into()),
                Operator::Neg => {
                    debug_assert!(args.len() == 1);

                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output_and_var_map(
                        workspace,
                        var_map,
                        var_name_map,
                        &mut base,
                    )?;

                    let num = workspace.new_num(-1);

                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();
                    mul.extend(base.as_view());
                    mul.extend(num.as_view());
                    mul_h.as_view().normalize(workspace, out);
                }
                Operator::Inv => {
                    debug_assert!(args.len() == 1);

                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output_and_var_map(
                        workspace,
                        var_map,
                        var_name_map,
                        &mut base,
                    )?;

                    let num = workspace.new_num(-1);

                    let mut pow_h = workspace.new_atom();
                    pow_h.to_pow(base.as_view(), num.as_view());
                    pow_h.as_view().normalize(workspace, out);
                }
            },
            Token::Fn(_, args) => {
                let name = match &args[0] {
                    Token::ID(s) => s,
                    _ => unreachable!(),
                };

                let index = var_name_map
                    .iter()
                    .position(|x| x == name)
                    .ok_or_else(|| format!("Undefined variable {}", name))?;
                if let Variable::Symbol(id) = var_map[index] {
                    let mut fun_h = workspace.new_atom();
                    let fun = fun_h.to_fun(id);
                    let mut atom = workspace.new_atom();
                    for a in args.iter().skip(1) {
                        a.to_atom_with_output_and_var_map(
                            workspace,
                            var_map,
                            var_name_map,
                            &mut atom,
                        )?;
                        fun.add_arg(atom.as_view());
                    }

                    fun_h.as_view().normalize(workspace, out);
                } else {
                    Err(format!("Undefined variable {}", name))?;
                }
            }
            x => return Err(format!("Unexpected token {}", x)),
        }

        Ok(())
    }

    /// Parse a Symbolica expression.
    pub fn parse(input: &str) -> Result<Token, String> {
        let mut stack: Vec<_> = Vec::with_capacity(20);
        stack.push(Token::Start);
        let mut state = ParseState::Any;

        let ops = ['\0', '^', '+', '*', '-', '(', ')', '/', ',', '[', ']'];
        let whitespace = [' ', '\t', '\n', '\r', '\\'];
        let forbidden = [';', ':', '&', '!', '%'];

        let mut char_iter = input.chars();
        let mut c = char_iter.next().unwrap_or('\0'); // add EOF as a token
        let mut extra_ops: SmallVec<[char; 6]> = SmallVec::new();

        let mut id_buffer = String::with_capacity(30);

        let mut line_counter = 1;
        let mut column_counter = 1;

        loop {
            match state {
                ParseState::Identifier => {
                    if ops.contains(&c) || whitespace.contains(&c) {
                        state = ParseState::Any;
                        stack.push(Token::ID(id_buffer.as_str().into()));
                        id_buffer.clear();
                    } else if !forbidden.contains(&c) {
                        id_buffer.push(c);
                    } else {
                        // check for some symbols that could be the result of copy-paste errors
                        // when importing from other languages
                        Err(format!(
                            "Unexpected '{}' in input at line {} and column {}",
                            c, line_counter, column_counter
                        ))?;
                    }
                }
                ParseState::Number => {
                    if c != '_' && c != ' ' && !c.is_ascii_digit() {
                        state = ParseState::Any;
                        stack.push(Token::Number(id_buffer.as_str().into()));
                        id_buffer.clear();
                    } else if c != '_' && c != ' ' {
                        id_buffer.push(c);
                    }
                }
                ParseState::RationalPolynomial => {
                    let start = char_iter.clone();
                    let mut pos = 0;

                    let mut s = SmartString::new();
                    s.push(c);

                    while c != ']' {
                        pos += 1;
                        c = char_iter.next().unwrap_or('\0');
                    }

                    s.push_str(&start.as_str()[..pos - 1]);
                    stack.push(Token::RationalPolynomial(s));

                    state = ParseState::Any;

                    column_counter += pos + 1;
                    c = char_iter.next().unwrap_or('\0');
                }
                ParseState::Any => {}
            }

            if state == ParseState::Any {
                if whitespace.contains(&c) {
                    if c == '\n' {
                        column_counter = 1;
                        line_counter += 1;
                    } else {
                        column_counter += 1;
                    }

                    c = char_iter.next().unwrap_or('\0');
                    continue;
                }

                match c {
                    '+' => {
                        if matches!(
                            unsafe { stack.last().unwrap_unchecked() },
                            Token::Start
                                | Token::OpenParenthesis
                                | Token::Fn(true, _)
                                | Token::Op(_, true, _, _)
                        ) {
                            // unary + operator, can be ignored as plus is the default
                        } else {
                            stack.push(Token::Op(true, true, Operator::Add, vec![]))
                        }
                    }
                    '^' => stack.push(Token::Op(true, true, Operator::Pow, vec![])),
                    '*' => stack.push(Token::Op(true, true, Operator::Mul, vec![])),
                    '-' => {
                        if matches!(
                            unsafe { stack.last().unwrap_unchecked() },
                            Token::Start
                                | Token::OpenParenthesis
                                | Token::Fn(true, _)
                                | Token::Op(_, true, _, _)
                        ) {
                            // unary minus only requires an argument to the right
                            stack.push(Token::Op(false, true, Operator::Neg, vec![]));
                        } else {
                            stack.push(Token::Op(true, true, Operator::Add, vec![]));
                            extra_ops.push('-'); // push a unary minus
                        }
                    }
                    '(' => {
                        // check if the opening bracket belongs to a function
                        if let Some(Token::ID(_)) = stack.last() {
                            let name = unsafe { stack.pop().unwrap_unchecked() };
                            if let Token::ID(_) = name {
                                stack.push(Token::Fn(true, vec![name])); // serves as open paren
                            }
                        } else if unsafe { stack.last().unwrap_unchecked() }.is_normal() {
                            // insert multiplication: x(...) -> x*(...)
                            stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                            extra_ops.push(c);
                        } else {
                            stack.push(Token::OpenParenthesis)
                        }
                    }
                    ')' => stack.push(Token::CloseParenthesis),
                    '/' => {
                        if matches!(
                            stack.last().unwrap(),
                            Token::Start
                                | Token::OpenParenthesis
                                | Token::Fn(true, _)
                                | Token::Op(_, true, _, _)
                        ) {
                            // unary inv only requires an argument to the right
                            stack.push(Token::Op(false, true, Operator::Inv, vec![]));
                        } else {
                            stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                            extra_ops.push('/'); // push a (unary) inverse
                        }
                    }
                    ',' => stack.push(Token::Op(true, true, Operator::Argument, vec![])),
                    '\0' => stack.push(Token::EOF),
                    '[' => {
                        if unsafe { stack.last().unwrap_unchecked() }.is_normal() {
                            // insert multiplication: x[3,4] -> x*[3,4]
                            stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                            extra_ops.push(c);
                        } else {
                            state = ParseState::RationalPolynomial;
                        }
                    }
                    _ => {
                        if unsafe { stack.last().unwrap_unchecked() }.is_normal() {
                            // insert multiplication: x y -> x*y
                            stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                            extra_ops.push(c);
                        } else if c.is_ascii_digit() {
                            state = ParseState::Number;
                            id_buffer.push(c);
                        } else if !forbidden.contains(&c) {
                            state = ParseState::Identifier;
                            id_buffer.push(c);
                        } else {
                            Err(format!(
                                "Unexpected '{}' in input at line {} and column {}",
                                c, line_counter, column_counter
                            ))?;
                        }
                    }
                }
            }

            // match on triplets of type operator identifier operator
            while state == ParseState::Any && stack.len() > 2 {
                if !unsafe { stack.get_unchecked(stack.len() - 2) }.is_normal() {
                    // check for the empty function

                    // check if the left operator needs a right-hand side and the new operator still needs a left-hand side
                    match unsafe { stack.get_unchecked(stack.len() - 1) } {
                        Token::Op(true, _, op, _) => {
                            Err(format!(
                            "Error at line {} and position {}: operator '{}' is missing left-hand side",
                            line_counter, column_counter, op,
                        ))?;
                        }

                        Token::CloseParenthesis => {
                            let pos = stack.len() - 2;
                            // check if we have an empty function
                            if let Token::Fn(f, _) = unsafe { stack.get_unchecked_mut(pos) } {
                                *f = false;
                                stack.pop();
                            } else {
                                Err(format!(
                                    "Error at line {} and position {}: unexpected ')'",
                                    line_counter, column_counter,
                                ))?;
                            }
                        }
                        _ => {}
                    }

                    // no simplification, get new token
                    break;
                }

                let mut last = unsafe { stack.pop().unwrap_unchecked() };
                let middle = unsafe { stack.pop().unwrap_unchecked() };
                let mut first = unsafe { stack.last_mut().unwrap_unchecked() };

                match first.get_precedence().cmp(&last.get_precedence()) {
                    std::cmp::Ordering::Greater => {
                        first.add_right(middle).map_err(|e| {
                            format!(
                                "Error at line {} and position {}: ",
                                line_counter, column_counter
                            ) + e.as_str()
                        })?;
                        stack.push(last);
                    }
                    std::cmp::Ordering::Less => {
                        last.add_left(middle).map_err(|e| {
                            format!(
                                "Error at line {} and position {}: ",
                                line_counter, column_counter
                            ) + e.as_str()
                        })?;

                        stack.push(last);
                    }
                    std::cmp::Ordering::Equal => {
                        // same degree, special merges!
                        match (&mut first, middle, last) {
                            (Token::Start, mid, Token::EOF) => {
                                *first = mid;
                            }
                            (Token::Fn(mr, args), mid, Token::CloseParenthesis) => {
                                debug_assert!(*mr);
                                *mr = false;

                                if let Token::Op(_, _, Operator::Argument, arg2) = mid {
                                    args.extend(arg2);
                                } else {
                                    args.push(mid);
                                }
                            }
                            (Token::OpenParenthesis, mid, Token::CloseParenthesis) => {
                                *first = mid;
                            }
                            (
                                Token::Op(ml1, mr1, o1, m),
                                mid,
                                Token::Op(ml2, mr2, mut o2, mut mm),
                            ) => {
                                debug_assert!(!*ml1);
                                debug_assert!(*mr1 && ml2);
                                // same precedence, so left associate

                                // flatten if middle identifier is also a binary operator of the same type that
                                // is also right associative
                                if let Token::Op(_, _, o_mid, mut m_mid) = mid {
                                    if o_mid == *o1 && o_mid.right_associative() {
                                        m.append(&mut m_mid);
                                    } else {
                                        m.push(Token::Op(false, false, o_mid, m_mid));
                                    }
                                } else {
                                    m.push(mid)
                                }

                                // may not be the same operator, in the case of * and /
                                if *o1 == o2 {
                                    m.append(&mut mm);
                                    *mr1 = mr2;
                                } else {
                                    // embed operator 1 in operator 2
                                    *mr1 = mr2;
                                    std::mem::swap(o1, &mut o2);
                                    std::mem::swap(m, &mut mm);
                                    m.insert(0, Token::Op(false, false, o2, mm));
                                }
                            }
                            _ => return Err("Cannot merge operator".to_string()),
                        }
                    }
                }
            }

            if c == '\0' {
                break;
            }

            // first drain the queue of extra operators
            if extra_ops.is_empty() {
                if c == '\n' {
                    column_counter = 1;
                    line_counter += 1;
                } else {
                    column_counter += 1;
                }

                c = char_iter.next().unwrap_or('\0');
            } else {
                c = extra_ops.remove(0);
            }
        }

        if stack.len() == 1 {
            Ok(stack.pop().unwrap())
        } else {
            match stack.get(stack.len() - 2) {
                Some(Token::Op(false, true, op, _)) => Err(format!(
                    "Unexpected end of input: missing right-hand side for operator '{}'",
                    op
                )),
                Some(Token::OpenParenthesis) => {
                    Err("Unexpected end of input: open parenthesis is not closed".to_string())
                }

                Some(Token::Fn(true, args)) => Err(format!(
                    "Unexpected end of input: Missing closing parenthesis for function '{}'",
                    args[0]
                )),
                Some(Token::Start) => Err("Expression is empty".to_string()),
                _ => Err(format!("Unknown parsing error: {:?}", stack)),
            }
        }
    }

    /// A special routine that can parse a polynomial written in expanded form,
    /// where the coefficient comes first.
    pub fn parse_polynomial<'a, R: Ring + ConvertToRing, E: Exponent>(
        mut input: &'a [u8],
        var_map: &Arc<Vec<Variable>>,
        var_name_map: &[SmartString<LazyCompact>],
        field: &R,
    ) -> (&'a [u8], MultivariatePolynomial<R, E>) {
        let mut exponents = vec![E::zero(); var_name_map.len()];
        let mut poly = MultivariatePolynomial::new(field, None, var_map.clone());

        let mut last_pos = input;
        let mut c = input.get_u8();

        let mut digit_buffer = vec![];
        loop {
            if c == b'(' || c == b')' || c == b'/' {
                break;
            }

            // read a term
            let mut coeff = field.one();
            for e in &mut exponents {
                *e = E::zero();
            }

            if c == b'+' {
                last_pos = input;
                c = input.get_u8();
            }

            // read number
            let num_start = last_pos;
            if c == b'-' {
                last_pos = input;
                c = input.get_u8();
            }

            let mut is_hex = false;
            let mask = if c == b'#' {
                is_hex = true;
                last_pos = input;
                c = input.get_u8();
                &HEX_DIGIT_MASK
            } else {
                &DIGIT_MASK
            };

            while mask[c as usize] && !input.is_empty() {
                last_pos = input;
                c = input.get_u8();
            }

            // construct number
            let mut len = unsafe { input.as_ptr().offset_from(num_start.as_ptr()) } as usize;
            let mut last_read_is_non_digit = false;
            if !mask[c as usize] && (len > 1 || c != b'-') {
                last_read_is_non_digit = true;
                len -= 1;
            }

            if len > 0 {
                coeff = 'read_coeff: {
                    if len == 1 && num_start[0] == b'-' {
                        break 'read_coeff field.neg(&field.one());
                    }

                    if !is_hex && len <= 40 {
                        let n = unsafe { std::str::from_utf8_unchecked(&num_start[..len]) };

                        if len <= 20 {
                            if let Ok(n) = n.parse::<i64>() {
                                break 'read_coeff field.element_from_coefficient(n.into());
                            }
                        }

                        if let Ok(n) = n.parse::<i128>() {
                            break 'read_coeff field
                                .element_from_coefficient(Integer::Double(n).into());
                        }
                    }

                    let (is_negative, digits) = if num_start[0] == b'-' {
                        (true, &num_start[1..len])
                    } else {
                        (false, &num_start[..len])
                    };

                    digit_buffer.clear();

                    if is_hex {
                        digit_buffer.extend(
                            digits[1..]
                                .iter()
                                .map(|&x| HEX_TO_DIGIT[(x - b'0') as usize]),
                        );
                    } else {
                        digit_buffer.extend(digits.iter().map(|&x| (x - b'0')));
                    }

                    let mut p = MultiPrecisionInteger::new();
                    unsafe {
                        p.assign_bytes_radix_unchecked(
                            &digit_buffer,
                            if is_hex { 16 } else { 10 },
                            is_negative,
                        )
                    };

                    field.element_from_coefficient(p.into())
                }
            }

            if input.is_empty() && !last_read_is_non_digit {
                poly.append_monomial(coeff, &exponents);
                break;
            }

            if c == b'-' {
                // done with the term
                poly.append_monomial(coeff, &exponents);
                continue;
            }

            // read var^pow
            loop {
                let before_star = last_pos;
                if c == b'*' {
                    if input.is_empty() {
                        break;
                    }

                    last_pos = input;
                    c = input.get_u8();
                }
                if !c.is_ascii_alphabetic() {
                    if before_star[0] == b'*' {
                        last_pos = before_star; // bring back the *
                    }
                    break;
                }

                let var_start = last_pos;

                // read var
                while c.is_ascii_alphanumeric() {
                    if input.is_empty() {
                        break;
                    }

                    last_pos = input;
                    c = input.get_u8();
                }

                let mut len = unsafe { input.as_ptr().offset_from(var_start.as_ptr()) } as usize;
                if !c.is_ascii_alphanumeric() {
                    len -= 1;
                }

                let name = unsafe { std::str::from_utf8_unchecked(&var_start[..len]) };
                let index = var_name_map
                    .iter()
                    .position(|x| x == name)
                    .expect("Undefined variable");

                // read pow
                if c == b'^' {
                    let pow_start = input;

                    // read pow
                    loop {
                        last_pos = input;
                        c = input.get_u8();

                        if !c.is_ascii_digit() || input.is_empty() {
                            break;
                        }
                    }

                    let mut len =
                        unsafe { input.as_ptr().offset_from(pow_start.as_ptr()) } as usize;
                    if !c.is_ascii_digit() {
                        len -= 1;
                    }
                    let n = unsafe { std::str::from_utf8_unchecked(&pow_start[..len]) };
                    exponents[index] = E::from_u32(n.parse::<u32>().unwrap());
                } else {
                    exponents[index] = E::one();
                }

                if input.is_empty() {
                    break;
                }
            }

            // contruct a new term
            poly.append_monomial(coeff, &exponents);

            if input.is_empty() {
                break;
            }
        }
        if input.is_empty() {
            (input, poly)
        } else {
            (last_pos, poly)
        }
    }
}
