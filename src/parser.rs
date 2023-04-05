use std::fmt::Write;

use rug::{Complete, Integer};

use crate::{
    representations::{number::Number, tree::AtomTree, Atom, OwnedAtom},
    state::State,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ParseState {
    Identifier(usize),
    Number(usize),
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOperator {
    Mul,
    Add,
    Pow,
    Div,
    Argument, // comma
}

impl BinaryOperator {
    fn get_precedence(&self) -> u8 {
        match self {
            BinaryOperator::Mul => 8,
            BinaryOperator::Div => 8,
            BinaryOperator::Add => 7,
            BinaryOperator::Pow => 9,
            BinaryOperator::Argument => 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Number(String),
    ID(String),
    BinaryOp(bool, bool, BinaryOperator, Vec<Token>),
    Fn(bool, String, Vec<Token>),
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
            Token::BinaryOp(more_left, more_right, _, a) => {
                a.len() > 1 && !more_left && !more_right
            }
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
                assert!(*ml);
                *ml = false;

                if let Token::BinaryOp(ml, mr, o2, mut args2) = other {
                    assert!(!ml && !mr);
                    if *o1 == o2 {
                        std::mem::swap(args, &mut args2);
                        args2.extend(args.drain(..));
                    } else {
                        args.insert(0, Token::BinaryOp(false, false, o2, args2));
                    }
                } else {
                    args.insert(0, other);
                }
            }
            _ => unreachable!(),
        }
    }

    /// Add `other` to right side of `self`, where `self` is a binary operation.
    fn add_right(&mut self, other: Token) {
        match self {
            Token::BinaryOp(_, mr, o1, args) => {
                assert!(*mr);
                *mr = false;

                if let Token::BinaryOp(ml, mr, o2, mut args2) = other {
                    assert!(!ml && !mr);
                    if *o1 == o2 {
                        args.extend(args2.drain(..))
                    } else {
                        args.push(Token::BinaryOp(false, false, o2, args2));
                    }
                } else {
                    args.push(other);
                }
            }
            _ => unreachable!(),
        }
    }

    pub fn to_atom_tree(self, state: &mut State) -> Result<AtomTree, String> {
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
                    BinaryOperator::Div => {
                        let base = atom_args.remove(0);
                        let inv_args = AtomTree::Mul(atom_args);

                        Ok(AtomTree::Mul(vec![
                            base,
                            AtomTree::Pow(Box::new((
                                inv_args,
                                AtomTree::Num(Number::Natural(-1, 1)),
                            ))),
                        ]))
                    }
                    BinaryOperator::Argument => Err(format!("Unexpected argument operator")),
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

    pub fn to_atom<P: Atom>(self, state: &mut State) -> Result<OwnedAtom<P>, String> {
        let a = self.to_atom_tree(state)?;
        Ok(P::from_tree(&a))
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
                            BinaryOperator::Div => f.write_char('/')?,
                            BinaryOperator::Add => f.write_char('+')?,
                            BinaryOperator::Pow => f.write_char('^')?,
                            BinaryOperator::Argument => f.write_char(',')?,
                        }
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
    let mut stack: Vec<Token> = vec![Token::Start];
    let mut state = ParseState::Any;

    let delims = ['\0', '^', '+', '*', '-', '(', ')', '/', ','];
    let whitespace = [' ', '\t', '\n', '\r', '\\'];

    let mut i = 0;
    loop {
        let c = input.chars().nth(i).unwrap_or('\0'); // add EOF as a token

        if whitespace.contains(&c) {
            i += 1;
            continue;
        }

        match state {
            ParseState::Identifier(token_start) => {
                if delims.contains(&c) {
                    state = ParseState::Any;
                    stack.push(Token::ID(
                        input[token_start..i]
                            .chars()
                            .filter(|a| !whitespace.contains(a))
                            .collect::<String>(),
                    ));
                }
            }
            ParseState::Number(token_start) => {
                if c != '_' && (c < '0' || c > '9') {
                    if !delims.contains(&c) {
                        return Err(format!(
                            "Parsing error at index {}. Unexpected continuation of number",
                            i
                        ));
                    }

                    // number is over
                    state = ParseState::Any;
                    stack.push(Token::Number(
                        input[token_start..i]
                            .chars()
                            .filter(|a| *a >= '0' && *a <= '9')
                            .collect::<String>(),
                    ));
                }
            }
            ParseState::Any => {}
        }

        if state == ParseState::Any {
            match c {
                '+' => {
                    if matches!(
                        stack.last().unwrap(),
                        Token::Start
                            | Token::OpenParenthesis
                            | Token::Fn(_, _, _)
                            | Token::BinaryOp(_, _, _, _)
                    ) {
                        // unary operator, can be ignored as plus is the default
                    } else {
                        stack.push(Token::BinaryOp(true, true, BinaryOperator::Add, vec![]))
                    }
                }
                '^' => stack.push(Token::BinaryOp(true, true, BinaryOperator::Pow, vec![])),
                '*' => stack.push(Token::BinaryOp(true, true, BinaryOperator::Mul, vec![])),
                '-' => {
                    if matches!(
                        stack.last().unwrap(),
                        Token::Start
                            | Token::OpenParenthesis
                            | Token::Fn(_, _, _)
                            | Token::BinaryOp(_, _, _, _)
                    ) {
                        stack.push(Token::BinaryOp(
                            false,
                            true,
                            BinaryOperator::Mul,
                            vec![Token::Number("-1".to_owned())],
                        ));
                    } else {
                        stack.push(Token::BinaryOp(true, true, BinaryOperator::Add, vec![]));
                        stack.push(Token::BinaryOp(
                            false,
                            true,
                            BinaryOperator::Mul,
                            vec![Token::Number("-1".to_owned())],
                        ));
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
                '/' => stack.push(Token::BinaryOp(true, true, BinaryOperator::Div, vec![])),
                ',' => stack.push(Token::BinaryOp(
                    true,
                    true,
                    BinaryOperator::Argument,
                    vec![],
                )),
                '\0' => stack.push(Token::EOF),
                x => {
                    if c >= '0' && c <= '9' {
                        state = ParseState::Number(i);
                    } else if c >= 'a' && c <= 'z' {
                        state = ParseState::Identifier(i);
                    } else {
                        return Err(format!("Unknown token {}", x));
                    }
                }
            }
        }

        // match on triplets of type operator identifier operator
        while stack.len() > 2 && state == ParseState::Any {
            let mut last = stack.pop().unwrap();
            let middle = stack.pop().unwrap();
            let mut first = stack.pop().unwrap();

            if !middle.is_normal() {
                // no simplification, get new token
                stack.push(first);
                stack.push(middle);
                stack.push(last);
                break;
            }

            match first.get_precedence().cmp(&last.get_precedence()) {
                std::cmp::Ordering::Greater => {
                    first.add_right(middle);
                    stack.push(first);
                    stack.push(last);
                }
                std::cmp::Ordering::Less => {
                    stack.push(first);
                    last.add_left(middle);
                    stack.push(last);
                }
                std::cmp::Ordering::Equal => {
                    // same degree, special merges!
                    let mut add_first = false;
                    match (&mut first, middle, last) {
                        (Token::Start, mid, Token::EOF) => {
                            stack.push(mid);
                        }
                        (Token::Fn(mr, _name, args), mid, Token::CloseParenthesis) => {
                            assert!(*mr);
                            *mr = false;

                            if let Token::BinaryOp(_, _, BinaryOperator::Argument, arg2) = mid {
                                args.extend(arg2);
                            } else {
                                args.push(mid);
                            }
                            add_first = true;
                        }
                        (Token::OpenParenthesis, mid, Token::CloseParenthesis) => {
                            stack.push(mid);
                        }
                        (
                            Token::BinaryOp(ml1, mr1, o1, m),
                            mid,
                            Token::BinaryOp(ml2, mr2, mut o2, mut mm),
                        ) => {
                            assert!(!*ml1);
                            assert!(*mr1 && ml2);
                            // same precedence, so left associate

                            // flatten if middle identifier is also a binary operator of the same type
                            if let Token::BinaryOp(_, _, o_mid, mut m_mid) = mid {
                                if o_mid == *o1 {
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

                            add_first = true;
                        }
                        _ => unreachable!(),
                    }

                    if add_first {
                        stack.push(first);
                    }
                }
            }
        }

        if c == '\0' {
            break;
        }

        i += 1;
    }

    if stack.len() == 1 {
        Ok(stack.pop().unwrap())
    } else {
        Err(format!("Parsing error: {:?}", stack))
    }
}
