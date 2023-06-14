use std::{fmt::Write, string::String};

use bytes::Buf;
use rug::{Complete, Integer};

use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

use crate::{
    poly::{polynomial::MultivariatePolynomial, Exponent},
    representations::{
        number::{ConvertToRing, Number},
        tree::AtomTree,
        Atom, Identifier, OwnedAtom,
    },
    rings::Ring,
    state::{State, Workspace},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ParseState {
    Identifier,
    Number,
    RationalPolynomial,
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operator {
    Mul,
    Add,
    Pow,
    Argument, // comma
    Neg,      // left side should be tagged as 'finished'
    Inv,      // left side should be tagged as 'finished', for internal use
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
            Operator::Argument => 5,
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

#[derive(Debug, Clone, PartialEq, Eq)]
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
    fn add_left(&mut self, other: Token) {
        match self {
            Token::Op(ml, _, o1, args) => {
                debug_assert!(*ml);
                *ml = false;

                if let Token::Op(ml, mr, o2, mut args2) = other {
                    debug_assert!(!ml && !mr);
                    if *o1 == o2 {
                        // add from the left by swapping and then extending from the right
                        std::mem::swap(args, &mut args2);
                        args.extend(args2.drain(..));
                    } else {
                        args.insert(0, Token::Op(false, false, o2, args2));
                    }
                } else {
                    args.insert(0, other);
                }
            }
            _ => unreachable!("Cannot left-append to non-operator"),
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
            _ => {
                let t = std::mem::replace(self, Token::EOF);
                *self = Token::Op(false, false, Operator::Neg, vec![t]);
            }
        }
    }

    /// Add `other` to right side of `self`, where `self` is a binary operation.
    #[inline]
    fn add_right(&mut self, mut other: Token) {
        match self {
            Token::Op(_, mr, o1, args) => {
                debug_assert!(*mr);
                *mr = false;

                if *o1 == Operator::Neg {
                    other.distribute_neg();
                    *self = other;
                    return;
                }

                if let Token::Op(ml, mr, o2, mut args2) = other {
                    debug_assert!(!ml && !mr);
                    if *o1 == o2 && o2.right_associative() {
                        if o2 == Operator::Neg || o2 == Operator::Inv {
                            // twice unary minus or inv cancels out
                            debug_assert!(args2.len() == 1);
                            *self = args2.pop().unwrap();
                        } else {
                            args.extend(args2.drain(..))
                        }
                    } else {
                        args.push(Token::Op(false, false, o2, args2));
                    }
                } else {
                    args.push(other);
                }
            }
            _ => unreachable!("Cannot right-append to non-operator"),
        }
    }

    pub fn to_atom_tree(&self, state: &mut State) -> Result<AtomTree, String> {
        match self {
            Token::Number(n) => {
                if let Ok(x) = n.parse::<i64>() {
                    Ok(AtomTree::Num(Number::Natural(x, 1)))
                } else {
                    match Integer::parse(n) {
                        Ok(x) => Ok(AtomTree::Num(Number::Large(x.complete().into()))),
                        Err(e) => Err(format!("Could not parse number: {}", e)),
                    }
                }
            }
            Token::ID(x) => Ok(AtomTree::Var(state.get_or_insert_var(x))),
            Token::Op(_, _, op, args) => {
                let mut atom_args = vec![];
                for a in args {
                    atom_args.push(a.to_atom_tree(state)?);
                }

                match op {
                    Operator::Mul => Ok(AtomTree::Mul(atom_args)),
                    Operator::Add => Ok(AtomTree::Add(atom_args)),
                    Operator::Pow => {
                        let base = atom_args.remove(0);
                        let exp = atom_args.remove(0);

                        let mut pow = AtomTree::Pow(Box::new((base, exp)));

                        for e in atom_args {
                            pow = AtomTree::Pow(Box::new((pow, e)));
                        }

                        Ok(pow)
                    }
                    Operator::Argument => Err("Unexpected argument operator".into()),
                    Operator::Neg => {
                        debug_assert!(atom_args.len() == 1);
                        Ok(AtomTree::Mul(vec![
                            atom_args.pop().unwrap(),
                            AtomTree::Num(Number::Natural(-1, 1)),
                        ]))
                    }
                    Operator::Inv => {
                        debug_assert!(atom_args.len() == 1);
                        Ok(AtomTree::Pow(Box::new((
                            atom_args.pop().unwrap(),
                            AtomTree::Num(Number::Natural(-1, 1)),
                        ))))
                    }
                }
            }
            Token::Fn(_, args) => {
                let name = match &args[0] {
                    Token::ID(s) => s,
                    _ => unreachable!(),
                };

                let mut atom_args = Vec::with_capacity(args.len() - 1);
                for a in args.iter().skip(1) {
                    atom_args.push(a.to_atom_tree(state)?);
                }
                Ok(AtomTree::Fn(state.get_or_insert_var(name), atom_args))
            }
            x => Err(format!("Unexpected token {}", x)),
        }
    }

    pub fn to_atom<P: Atom>(
        &self,
        state: &mut State,
        workspace: &Workspace<P>,
    ) -> Result<OwnedAtom<P>, String> {
        let a = self.to_atom_tree(state)?;
        Ok(P::from_tree(&a, state, workspace))
    }
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
                    Token::ID(s) => f.write_str(&s)?,
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
            _ => unreachable!(),
        }
    }
}

pub fn parse(input: &str) -> Result<Token, String> {
    let mut stack: Vec<_> = Vec::with_capacity(20);
    stack.push(Token::Start);
    let mut state = ParseState::Any;

    let delims = ['\0', '^', '+', '*', '-', '(', ')', '/', ',', '[', ']'];
    let whitespace = [' ', '\t', '\n', '\r', '\\'];

    let mut char_iter = input.chars();
    let mut c = char_iter.next().unwrap_or('\0'); // add EOF as a token
    let mut extra_ops: SmallVec<[char; 6]> = SmallVec::new();

    let mut id_buffer = String::with_capacity(30);

    let mut i = 0;
    loop {
        if whitespace.contains(&c) {
            i += 1;
            c = char_iter.next().unwrap_or('\0');
            continue;
        }

        match state {
            ParseState::Identifier => {
                if delims.contains(&c) {
                    state = ParseState::Any;
                    stack.push(Token::ID(id_buffer.as_str().into()));
                    id_buffer.clear();
                } else {
                    id_buffer.push(c);
                }
            }
            ParseState::Number => {
                if c != '_' && !c.is_ascii_digit() {
                    if !delims.contains(&c) {
                        return Err(format!(
                            "Parsing error at index {}. Unexpected continuation of number",
                            i
                        ));
                    }

                    // drag in the neg operator
                    if state == ParseState::Any {
                        if let Some(Token::Op(false, true, Operator::Neg, _)) = stack.last_mut() {
                            stack.pop();
                            id_buffer.insert(0, '-');
                        }
                    }

                    state = ParseState::Any;

                    stack.push(Token::Number(id_buffer.as_str().into()));

                    id_buffer.clear();
                } else {
                    id_buffer.push(c);
                }
            }
            ParseState::RationalPolynomial => {
                if c == ']' {
                    stack.push(Token::RationalPolynomial(id_buffer.as_str().into()));
                    id_buffer.clear();

                    state = ParseState::Any;
                    i += 1;
                    c = char_iter.next().unwrap_or('\0');
                    continue; // whitespace may have to be skipped
                } else {
                    id_buffer.push(c);
                }
            }
            ParseState::Any => {}
        }

        if state == ParseState::Any {
            if !c.is_ascii() {
                state = ParseState::Identifier;
                id_buffer.push(c);
            }

            match c as u8 {
                b'+' => {
                    if matches!(
                        unsafe { stack.last().unwrap_unchecked() },
                        Token::Start | Token::OpenParenthesis | Token::Op(_, true, _, _)
                    ) {
                        // unary operator, can be ignored as plus is the default
                    } else {
                        stack.push(Token::Op(true, true, Operator::Add, vec![]))
                    }
                }
                b'^' => stack.push(Token::Op(true, true, Operator::Pow, vec![])),
                b'*' => stack.push(Token::Op(true, true, Operator::Mul, vec![])),
                b'-' => {
                    if matches!(
                        unsafe { stack.last().unwrap_unchecked() },
                        Token::Start | Token::OpenParenthesis | Token::Op(_, true, _, _)
                    ) {
                        // unary minus only requires an argument to the right
                        stack.push(Token::Op(false, true, Operator::Neg, vec![]));
                    } else {
                        stack.push(Token::Op(true, true, Operator::Add, vec![]));
                        extra_ops.push('-'); // push a unary minus
                    }
                }
                b'(' => {
                    // check if the opening bracket belongs to a function
                    if let Some(Token::ID(_)) = stack.last() {
                        let name = unsafe { stack.pop().unwrap_unchecked() };
                        if let Token::ID(_) = name {
                            stack.push(Token::Fn(true, vec![name])); // serves as open paren
                        }
                    } else {
                        // TODO: crash when a number if written before it
                        stack.push(Token::OpenParenthesis)
                    }
                }
                b')' => stack.push(Token::CloseParenthesis),
                b'/' => {
                    if matches!(
                        stack.last().unwrap(),
                        Token::Start | Token::OpenParenthesis | Token::Op(_, true, _, _)
                    ) {
                        // unary inv only requires an argument to the right
                        stack.push(Token::Op(false, true, Operator::Inv, vec![]));
                    } else {
                        stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                        extra_ops.push('/'); // push a (unary) inverse
                    }
                }
                b',' => stack.push(Token::Op(true, true, Operator::Argument, vec![])),
                b'\0' => stack.push(Token::EOF),
                b'[' => {
                    state = ParseState::RationalPolynomial;
                }
                x => {
                    if x >= b'0' && x <= b'9' {
                        state = ParseState::Number;
                        id_buffer.push(c);
                    } else {
                        state = ParseState::Identifier;
                        id_buffer.push(c);
                    }
                }
            }
        }

        // match on triplets of type operator identifier operator
        while state == ParseState::Any && stack.len() > 2 {
            if !unsafe { stack.get_unchecked(stack.len() - 2) }.is_normal() {
                // no simplification, get new token
                break;
            }

            let mut last = unsafe { stack.pop().unwrap_unchecked() };
            let middle = unsafe { stack.pop().unwrap_unchecked() };
            let mut first = unsafe { stack.last_mut().unwrap_unchecked() };

            match first.get_precedence().cmp(&last.get_precedence()) {
                std::cmp::Ordering::Greater => {
                    first.add_right(middle);
                    stack.push(last);
                }
                std::cmp::Ordering::Less => {
                    last.add_left(middle);
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
                        (Token::Op(ml1, mr1, o1, m), mid, Token::Op(ml2, mr2, mut o2, mut mm)) => {
                            debug_assert!(!*ml1);
                            debug_assert!(*mr1 && ml2);
                            // same precedence, so left associate

                            // flatten if middle identifier is also a binary operator of the same type that
                            // is also right associative
                            if let Token::Op(_, _, o_mid, mut m_mid) = mid {
                                if o_mid == *o1 && o_mid.right_associative() {
                                    m.extend(m_mid.drain(..));
                                } else {
                                    m.push(Token::Op(false, false, o_mid, m_mid));
                                }
                            } else {
                                m.push(mid)
                            }

                            // may not be the same operator, in the case of * and /
                            if *o1 == o2 {
                                m.extend(mm.drain(..));
                                *mr1 = mr2;
                            } else {
                                // embed operator 1 in operator 2
                                *mr1 = mr2;
                                std::mem::swap(o1, &mut o2);
                                std::mem::swap(m, &mut mm);
                                m.insert(0, Token::Op(false, false, o2, mm));
                            }
                        }
                        _ => return Err(format!("Cannot merge operator")),
                    }
                }
            }
        }

        if c == '\0' {
            break;
        }

        // first drain the queue of extra operators
        if extra_ops.is_empty() {
            i += 1;
            c = char_iter.next().unwrap_or('\0');
        } else {
            c = extra_ops.remove(0);
        }
    }

    if stack.len() == 1 {
        Ok(stack.pop().unwrap())
    } else {
        Err(format!("Parsing error: {:?}", stack))
    }
}

/// A special routine that can parse a polynomial written in expanded form,
/// where the coefficient comes first.
pub fn parse_polynomial<'a, R: Ring + ConvertToRing, E: Exponent>(
    mut input: &'a [u8],
    var_map: &[Identifier],
    var_name_map: &[SmartString<LazyCompact>],
    field: R,
) -> (&'a [u8], MultivariatePolynomial<R, E>) {
    let mut exponents = vec![E::zero(); var_name_map.len()];
    let mut poly = MultivariatePolynomial::new(
        var_name_map.len(),
        field.clone(),
        None,
        Some(var_map.into()),
    );

    let mut last_pos = input;
    let mut c = input.get_u8();
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

        loop {
            if !c.is_ascii_digit() {
                break;
            }

            if input.len() == 0 {
                break;
            }

            last_pos = input;
            c = input.get_u8();
        }

        // construct number
        let mut len = unsafe { input.as_ptr().offset_from(num_start.as_ptr()) } as usize;
        if !c.is_ascii_digit() && (len > 1 || c != b'-') {
            len -= 1;
        }

        if len > 0 {
            let n = unsafe { std::str::from_utf8_unchecked(&num_start[..len]) };

            if len == 1 && num_start[0] == b'-' {
                coeff = field.neg(&field.one());
            } else {
                coeff = if let Ok(x) = n.parse::<i64>() {
                    field.from_number(Number::Natural(x, 1))
                } else {
                    match Integer::parse(n) {
                        Ok(x) => {
                            let p = x.complete().into();
                            field.from_number(Number::Large(p))
                        }
                        Err(e) => panic!("Could not parse number: {}", e),
                    }
                };
            }
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
                if input.len() == 0 {
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
                if input.len() == 0 {
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

                    if !c.is_ascii_digit() || input.len() == 0 {
                        break;
                    }
                }

                let mut len = unsafe { input.as_ptr().offset_from(pow_start.as_ptr()) } as usize;
                if !c.is_ascii_digit() {
                    len -= 1;
                }
                let n = unsafe { std::str::from_utf8_unchecked(&pow_start[..len]) };
                exponents[index] = E::from_u32(n.parse::<u32>().unwrap());
            } else {
                exponents[index] = E::one();
            }

            if input.len() == 0 {
                break;
            }
        }

        // contruct a new term
        poly.append_monomial(coeff, &exponents);

        if input.len() == 0 {
            break;
        }
    }
    if input.len() == 0 {
        (input, poly)
    } else {
        (last_pos, poly)
    }
}
