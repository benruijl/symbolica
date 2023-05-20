use std::{fmt::Write, string::String};

use rug::{Complete, Integer};

use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

use crate::{
    representations::{number::Number, tree::AtomTree, Atom, OwnedAtom},
    state::{State, Workspace},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ParseState {
    Identifier,
    Number,
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOperator {
    Mul,
    Add,
    Pow,
    Argument, // comma
    Neg,      // left side should be tagged as 'finished'
    Inv,      // left side should be tagged as 'finished', for internal use
}

impl BinaryOperator {
    fn get_precedence(&self) -> u8 {
        match self {
            BinaryOperator::Mul => 8,
            BinaryOperator::Add => 7,
            BinaryOperator::Pow => 11,
            BinaryOperator::Argument => 5,
            BinaryOperator::Neg => 10,
            BinaryOperator::Inv => 9,
        }
    }

    fn right_associative(&self) -> bool {
        match self {
            BinaryOperator::Mul => true,
            BinaryOperator::Add => true,
            BinaryOperator::Pow => false,
            BinaryOperator::Argument => true,
            BinaryOperator::Neg => true,
            BinaryOperator::Inv => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Number(SmartString<LazyCompact>),
    ID(SmartString<LazyCompact>),
    BinaryOp(bool, bool, BinaryOperator, Vec<Token>),
    Fn(bool, SmartString<LazyCompact>, Vec<Token>),
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
            Token::BinaryOp(more_left, more_right, _, _) => !more_left && !more_right,
            Token::Fn(more_right, _, _) => !more_right,
            _ => false,
        }
    }

    /// Get the precedence of the token.
    fn get_precedence(&self) -> u8 {
        match self {
            Token::Number(_) => 10,
            Token::ID(_) => 10,
            Token::BinaryOp(_, _, o, _) => o.get_precedence(),
            Token::Fn(_, _, _) | Token::OpenParenthesis | Token::CloseParenthesis => 5,
            Token::Start | Token::EOF => 4,
        }
    }

    /// Add `other` to the left side of `self`, where `self` is a binary operation.
    fn add_left(&mut self, other: Token) {
        match self {
            Token::BinaryOp(ml, _, o1, args) => {
                debug_assert!(*ml);
                *ml = false;

                if let Token::BinaryOp(ml, mr, o2, mut args2) = other {
                    debug_assert!(!ml && !mr);
                    if *o1 == o2 {
                        // add from the left by swapping and then extending from the right
                        std::mem::swap(args, &mut args2);
                        args.extend(args2.drain(..));
                    } else {
                        args.insert(0, Token::BinaryOp(false, false, o2, args2));
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
            Token::BinaryOp(_, _, BinaryOperator::Neg, args3) => {
                debug_assert!(args3.len() == 1);
                *self = args3.pop().unwrap();
            }
            Token::BinaryOp(_, _, BinaryOperator::Mul, args2) => {
                args2[0].distribute_neg();
            }
            Token::BinaryOp(_, _, BinaryOperator::Add, args2) => {
                for a in args2 {
                    a.distribute_neg();
                }
            }
            _ => {
                let t = std::mem::replace(self, Token::EOF);
                *self = Token::BinaryOp(false, false, BinaryOperator::Neg, vec![t]);
            }
        }
    }

    /// Add `other` to right side of `self`, where `self` is a binary operation.
    fn add_right(&mut self, mut other: Token) {
        match self {
            Token::BinaryOp(_, mr, o1, args) => {
                debug_assert!(*mr);
                *mr = false;

                if *o1 == BinaryOperator::Neg {
                    other.distribute_neg();
                    *self = other;
                    return;
                }

                if let Token::BinaryOp(ml, mr, o2, mut args2) = other {
                    debug_assert!(!ml && !mr);
                    if *o1 == o2 && o2.right_associative() {
                        if o2 == BinaryOperator::Neg || o2 == BinaryOperator::Inv {
                            // twice unary minus or inv cancels out
                            debug_assert!(args2.len() == 1);
                            *self = args2.pop().unwrap();
                        } else {
                            args.extend(args2.drain(..))
                        }
                    } else {
                        args.push(Token::BinaryOp(false, false, o2, args2));
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
            Token::BinaryOp(_, _, op, args) => {
                let mut atom_args = vec![];
                for a in args {
                    atom_args.push(a.to_atom_tree(state)?);
                }

                match op {
                    BinaryOperator::Mul => Ok(AtomTree::Mul(atom_args)),
                    BinaryOperator::Add => Ok(AtomTree::Add(atom_args)),
                    BinaryOperator::Pow => {
                        let base = atom_args.remove(0);
                        let exp = atom_args.remove(0);

                        let mut pow = AtomTree::Pow(Box::new((base, exp)));

                        for e in atom_args {
                            pow = AtomTree::Pow(Box::new((pow, e)));
                        }

                        Ok(pow)
                    }
                    BinaryOperator::Argument => Err("Unexpected argument operator".into()),
                    BinaryOperator::Neg => {
                        debug_assert!(atom_args.len() == 1);
                        Ok(AtomTree::Mul(vec![
                            atom_args.pop().unwrap(),
                            AtomTree::Num(Number::Natural(-1, 1)),
                        ]))
                    }
                    BinaryOperator::Inv => {
                        debug_assert!(atom_args.len() == 1);
                        Ok(AtomTree::Pow(Box::new((
                            atom_args.pop().unwrap(),
                            AtomTree::Num(Number::Natural(-1, 1)),
                        ))))
                    }
                }
            }
            Token::Fn(_, name, args) => {
                let mut atom_args = vec![];
                for a in args {
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
            Token::BinaryOp(_, _, o, m) => {
                let mut first = true;
                f.write_char('(')?;

                for mm in m {
                    if !first {
                        match o {
                            BinaryOperator::Mul => f.write_char('*')?,
                            BinaryOperator::Add => f.write_char('+')?,
                            BinaryOperator::Pow => f.write_char('^')?,
                            BinaryOperator::Argument => f.write_char(',')?,
                            BinaryOperator::Neg => f.write_char('-')?,
                            BinaryOperator::Inv => f.write_str("1/")?,
                        }
                    } else if *o == BinaryOperator::Neg {
                        f.write_char('-')?;
                    } else if *o == BinaryOperator::Inv {
                        f.write_str("1/")?;
                    }
                    first = false;

                    mm.fmt(f)?;
                }
                f.write_char(')')
            }
            Token::Fn(_, name, args) => {
                let mut first = true;

                f.write_str(name)?;

                f.write_char('(')?;
                for aa in args {
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

    let delims = ['\0', '^', '+', '*', '-', '(', ')', '/', ','];
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
                if c != '_' && (c < '0' || c > '9') {
                    if !delims.contains(&c) {
                        return Err(format!(
                            "Parsing error at index {}. Unexpected continuation of number",
                            i
                        ));
                    }

                    // number is over
                    state = ParseState::Any;

                    // drag in the neg operator
                    if let Some(Token::BinaryOp(false, true, BinaryOperator::Neg, _)) =
                        stack.last_mut()
                    {
                        stack.pop();
                        id_buffer.insert(0, '-');
                    }

                    stack.push(Token::Number(id_buffer.as_str().into()));
                    id_buffer.clear();
                } else {
                    id_buffer.push(c);
                }
            }
            ParseState::Any => {}
        }

        if state == ParseState::Any {
            match c {
                '+' => {
                    if matches!(
                        stack.last().unwrap(),
                        Token::Start | Token::OpenParenthesis | Token::BinaryOp(_, true, _, _)
                    ) {
                        // unary operator, can be ignored as plus is the default
                    } else {
                        stack.push(Token::BinaryOp(
                            true,
                            true,
                            BinaryOperator::Add,
                            Vec::with_capacity(2),
                        ))
                    }
                }
                '^' => stack.push(Token::BinaryOp(
                    true,
                    true,
                    BinaryOperator::Pow,
                    Vec::with_capacity(2),
                )),
                '*' => stack.push(Token::BinaryOp(
                    true,
                    true,
                    BinaryOperator::Mul,
                    Vec::with_capacity(2),
                )),
                '-' => {
                    if matches!(
                        stack.last().unwrap(),
                        Token::Start | Token::OpenParenthesis | Token::BinaryOp(_, true, _, _)
                    ) {
                        // unary minus only requires an argument to the right
                        stack.push(Token::BinaryOp(
                            false,
                            true,
                            BinaryOperator::Neg,
                            Vec::with_capacity(1),
                        ));
                    } else {
                        stack.push(Token::BinaryOp(
                            true,
                            true,
                            BinaryOperator::Add,
                            Vec::with_capacity(2),
                        ));
                        extra_ops.push('-'); // push a unary minus
                    }
                }
                '(' => {
                    // check if the opening bracket belongs to a function
                    if let Some(Token::ID(_)) = stack.last() {
                        let name = stack.pop().unwrap();
                        if let Token::ID(name) = name {
                            stack.push(Token::Fn(true, name, vec![])); // serves as open paren
                        }
                    } else {
                        // TODO: crash when a number if written before it
                        stack.push(Token::OpenParenthesis)
                    }
                }
                ')' => stack.push(Token::CloseParenthesis),
                '/' => {
                    if matches!(
                        stack.last().unwrap(),
                        Token::Start | Token::OpenParenthesis | Token::BinaryOp(_, true, _, _)
                    ) {
                        // unary inv only requires an argument to the right
                        stack.push(Token::BinaryOp(
                            false,
                            true,
                            BinaryOperator::Inv,
                            Vec::with_capacity(1),
                        ));
                    } else {
                        stack.push(Token::BinaryOp(
                            true,
                            true,
                            BinaryOperator::Mul,
                            Vec::with_capacity(2),
                        ));
                        extra_ops.push('/'); // push a (unary) inverse
                    }
                }
                ',' => stack.push(Token::BinaryOp(
                    true,
                    true,
                    BinaryOperator::Argument,
                    vec![],
                )),
                '\0' => stack.push(Token::EOF),
                x => {
                    if c >= '0' && c <= '9' {
                        state = ParseState::Number;
                        id_buffer.push(c);
                    } else if c >= 'a' && c <= 'z' {
                        state = ParseState::Identifier;
                        id_buffer.push(c);
                    } else {
                        return Err(format!("Unknown token {}", x));
                    }
                }
            }
        }

        // match on triplets of type operator identifier operator
        while stack.len() > 2 && state == ParseState::Any {
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
                        (Token::Fn(mr, _name, args), mid, Token::CloseParenthesis) => {
                            debug_assert!(*mr);
                            *mr = false;

                            if let Token::BinaryOp(_, _, BinaryOperator::Argument, arg2) = mid {
                                args.extend(arg2);
                            } else {
                                args.push(mid);
                            }
                        }
                        (Token::OpenParenthesis, mid, Token::CloseParenthesis) => {
                            *first = mid;
                        }
                        (
                            Token::BinaryOp(ml1, mr1, o1, m),
                            mid,
                            Token::BinaryOp(ml2, mr2, mut o2, mut mm),
                        ) => {
                            debug_assert!(!*ml1);
                            debug_assert!(*mr1 && ml2);
                            // same precedence, so left associate

                            // flatten if middle identifier is also a binary operator of the same type that
                            // is also right associative
                            if let Token::BinaryOp(_, _, o_mid, mut m_mid) = mid {
                                if o_mid == *o1 && o_mid.right_associative() {
                                    m.extend(m_mid.drain(..));
                                } else {
                                    m.push(Token::BinaryOp(false, false, o_mid, m_mid));
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
                                m.insert(0, Token::BinaryOp(false, false, o2, mm));
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
        if !extra_ops.is_empty() {
            c = extra_ops.remove(0);
        } else {
            i += 1;
            c = char_iter.next().unwrap_or('\0');
        }
    }

    if stack.len() == 1 {
        Ok(stack.pop().unwrap())
    } else {
        Err(format!("Parsing error: {:?}", stack))
    }
}
