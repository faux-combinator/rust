use std::iter;

pub struct Token<'a, TT> {
    token_type: TT,
    token_data: &'a str,
}

#[derive(PartialEq, Eq, Debug)]
pub enum ParserError<TT> {
    Eof,
    Either(Box<(ParserError<TT>, ParserError<TT>)>),
    Got(TT),
}

impl<'a, TT: Eq + Copy> Token<'a, TT> {
    pub fn new(token_type: TT, token_data: &'a str) -> Self {
        Token { token_type, token_data }
    }
}

pub struct Parser<'a, TT: Eq + Copy> {
    tokens: &'a Vec<Token<'a, TT>>,
    idx: usize,
}

impl<'a, TT: Eq + Copy> Parser<'a, TT> {
    pub fn new(tokens: &'a Vec<Token<'a, TT>>) -> Self {
        Parser { tokens, idx: 0 }
    }

    pub fn is_eof(&self) -> bool {
        self.idx >= self.tokens.len()
    }

    fn check_eof(&self) -> Result<(), ParserError<TT>> {
        if self.is_eof() {
            Err(ParserError::Eof)
        } else {
            Ok(())
        }
    }

    pub fn peek(&self) -> Result<&'a str, ParserError<TT>> {
        self.check_eof()?;
        Ok(self.tokens[self.idx].token_data)
    }

    pub fn expect(&mut self, expected: TT) -> Result<&'a str, ParserError<TT>> {
        self.check_eof()?;
        let next = &self.tokens[self.idx];
        if next.token_type == expected {
            self.idx += 1;
            Ok(next.token_data)
        } else {
            Err(ParserError::Got(next.token_type))
        }
    }

    pub fn attempt<T, F>(&mut self, func: F) -> Result<T, ParserError<TT>>
        where F: FnOnce(&mut Self) -> Result<T, ParserError<TT>>
    {
        let cur = self.idx;
        match func(self) {
            Ok(r) => Ok(r),
            Err(r) => {
                self.idx = cur;
                Err(r)
            }
        }
    }

    pub fn either<T, F1, F2>(&mut self, fn1: F1, fn2: F2) -> Result<T, ParserError<TT>>
        where F1: FnOnce(&mut Self) -> Result<T, ParserError<TT>>,
              F2: FnOnce(&mut Self) -> Result<T, ParserError<TT>>
    {
        match fn1(self) {
            Ok(r) => Ok(r),
            Err(e1) => match fn2(self) {
                Ok(r) => Ok(r),
                Err(e2) => Err(ParserError::Either(Box::new((e1, e2))))
            }
        }
    }

    pub fn any<T, F>(&mut self, mut func: F) -> Vec<T>
        where F: FnMut(&mut Self) -> Result<T, ParserError<TT>>
    {
        iter::from_fn(|| func(self).ok()).collect()
    }

    pub fn many<T, F>(&mut self, mut func: F) -> Result<Vec<T>, ParserError<TT>>
        where F: FnMut(&mut Self) -> Result<T, ParserError<TT>>
    {
        let fst = func(self)?;
        let mut xs: Vec<T> = iter::from_fn(|| func(self).ok()).collect();
        xs.splice(0..0, vec![fst].drain(..));
        Ok(xs)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use super::*;

    #[derive(Eq, PartialEq, Debug, Copy, Clone)]
    enum TokenType { Lparen, Id, Rparen }

    #[test]
    fn eof_true() {
        let tokens: Vec<Token<TokenType>> = vec![];
        let parser = Parser::new(&tokens);
        assert!(parser.is_eof());
    }

    #[test]
    fn eof_false() {
        let tokens = vec![Token::new(TokenType::Lparen, "(")];
        let parser = Parser::new(&tokens);
        assert!(!parser.is_eof());
    }

    #[test]
    fn peek_eof() {
        let tokens: Vec<Token<TokenType>> = vec![];
        let parser = Parser::new(&tokens);
        assert_eq!(parser.peek(), Err(ParserError::Eof));
    }

    #[test]
    fn peek_ok() {
        let tokens: Vec<Token<TokenType>> = vec![Token::new(TokenType::Lparen, "(")];
        let parser = Parser::new(&tokens);
        assert_eq!(parser.peek(), Ok("("));
    }

    #[test]
    fn expect_eof() {
        let tokens: Vec<Token<TokenType>> = vec![];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Err(ParserError::Eof));
    }

    #[test]
    fn expect_ko() {
        let tokens: Vec<Token<TokenType>> = vec![Token::new(TokenType::Lparen, "(")];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Rparen), Err(ParserError::Got(TokenType::Lparen)));
    }

    #[test]
    fn expect_ok() {
        let tokens: Vec<Token<TokenType>> = vec![Token::new(TokenType::Lparen, "(")];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
    }

    #[test]
    fn expect_ok_eof() {
        let tokens: Vec<Token<TokenType>> = vec![Token::new(TokenType::Lparen, "(")];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        assert_eq!(parser.expect(TokenType::Rparen), Err(ParserError::Eof));
    }

    #[test]
    fn expect_ok_ko() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        assert_eq!(parser.expect(TokenType::Lparen), Err(ParserError::Got(TokenType::Rparen)));
    }

    #[test]
    fn expect_ok_ok() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        assert_eq!(parser.expect(TokenType::Rparen), Ok(")"));
    }

    #[test]
    fn attempt_ok() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        let r = parser.attempt(|parser| {
            parser.expect(TokenType::Lparen)?;
            parser.expect(TokenType::Rparen)?;
            Ok(1)
        });
        assert_eq!(r, Ok(1));
        assert!(parser.is_eof());
    }

    #[test]
    fn attempt_ko() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Rparen, ")"),
            Token::new(TokenType::Lparen, "("),
        ];
        let mut parser = Parser::new(&tokens);
        let r = parser.attempt(|parser| {
            parser.expect(TokenType::Lparen)?;
            parser.expect(TokenType::Rparen)?;
            Ok(1)
        });
        assert_eq!(r, Err(ParserError::Got(TokenType::Rparen)));
        assert_eq!(parser.peek(), Ok(")"));
    }

    #[test]
    fn attempt_ko_backtracks() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Lparen, "("),
        ];
        let mut parser = Parser::new(&tokens);
        let r = parser.attempt(|parser| {
            parser.expect(TokenType::Lparen)?;
            parser.expect(TokenType::Rparen)?;
            Ok(1)
        });
        assert_eq!(r, Err(ParserError::Got(TokenType::Lparen)));
        assert_eq!(parser.peek(), Ok("("));
    }

    #[test]
    fn either_ko() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Id, "a"),
        ];
        let mut parser = Parser::new(&tokens);
        match parser.either(
            |parser: &mut Parser<TokenType>| parser.expect(TokenType::Lparen),
            |parser: &mut Parser<TokenType>| parser.expect(TokenType::Rparen),
        ) {
            Ok(_) => panic!("Unmatched either didn't error"),
            Err(r) => {
                match r {
                    ParserError::Either(errs) => {
                        let (a, b) = *errs;
                        assert_eq!(a, ParserError::Got(TokenType::Id));
                        assert_eq!(b, ParserError::Got(TokenType::Id));
                    }
                    _ => panic!("Either error isn't ParserError::Either")
                }
            }
        };
    }

    #[test]
    fn either_first() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
        ];
        let mut parser = Parser::new(&tokens);
        match parser.either(
            |parser: &mut Parser<TokenType>| parser.expect(TokenType::Lparen),
            |parser: &mut Parser<TokenType>| parser.expect(TokenType::Rparen),
        ) {
            Ok(r) => assert_eq!(r, "("),
            Err(_) => panic!("either didn't match")
        }
    }

    #[test]
    fn either_last() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        match parser.either(
            |parser: &mut Parser<TokenType>| parser.expect(TokenType::Lparen),
            |parser: &mut Parser<TokenType>| parser.expect(TokenType::Rparen),
        ) {
            Ok(r) => assert_eq!(r, ")"),
            Err(_) => panic!("either didn't match")
        }
    }

    #[test]
    fn any_zero() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        let ids = parser.any(|parser| parser.expect(TokenType::Id));
        assert_eq!(ids.len(), 0);
        assert_eq!(parser.expect(TokenType::Rparen), Ok(")"));
    }

    #[test]
    fn any_one() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Id, "a"),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        let ids = parser.any(|parser| parser.expect(TokenType::Id));
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], "a");
        assert_eq!(parser.expect(TokenType::Rparen), Ok(")"));
    }

    #[test]
    fn any_many() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Id, "a"),
            Token::new(TokenType::Id, "b"),
            Token::new(TokenType::Id, "c"),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        let ids = parser.any(|parser| parser.expect(TokenType::Id));
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], "a");
        assert_eq!(ids[1], "b");
        assert_eq!(ids[2], "c");
        assert_eq!(parser.expect(TokenType::Rparen), Ok(")"));
    }

    #[test]
    fn many_zero() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        match parser.many(|parser| parser.expect(TokenType::Id)) {
            Err(r) => assert_eq!(r, ParserError::Got(TokenType::Rparen)),
            Ok(_) => panic!("Unmatched many didn't error")
        }
    }

    #[test]
    fn many_one() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Id, "a"),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        match parser.many(|parser| parser.expect(TokenType::Id)) {
            Ok(ids) => {
                assert_eq!(ids.len(), 1);
                assert_eq!(ids[0], "a");
            }
            Err(_) => panic!("Many one errored")
        }
        assert_eq!(parser.expect(TokenType::Rparen), Ok(")"));
    }

    #[test]
    fn many_many() {
        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Id, "a"),
            Token::new(TokenType::Id, "b"),
            Token::new(TokenType::Id, "c"),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.expect(TokenType::Lparen), Ok("("));
        match parser.many(|parser| parser.expect(TokenType::Id)) {
            Ok(ids) => {
                assert_eq!(ids.len(), 3);
                assert_eq!(ids[0], "a");
                assert_eq!(ids[1], "b");
                assert_eq!(ids[2], "c");
            }
            Err(_) => panic!("Many many errored")
        }
        assert_eq!(parser.expect(TokenType::Rparen), Ok(")"));
    }

    #[test]
    fn tree() {
        enum Expr<'a> {
            Id(&'a str),
            Call(Box<Expr<'a>>, Vec<Expr<'a>>),
        }
        impl<'a> fmt::Display for Expr<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self {
                    Expr::Id(x) => write!(f, "{}", x),
                    Expr::Call(callee, args) => {
                        callee.fmt(f)?;
                        write!(f, "(")?;
                        for (i, el) in args.iter().enumerate() {
                            if i != 0 {
                                write!(f, ", ")?;
                            }
                            el.fmt(f)?;
                        }
                        write!(f, ")")
                    }
                }
            }
        }

        let tokens: Vec<Token<TokenType>> = vec![
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Id, "a"),
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Id, "b"),
            Token::new(TokenType::Rparen, ")"),
            Token::new(TokenType::Rparen, ")"),
            Token::new(TokenType::Lparen, "("),
            Token::new(TokenType::Id, "c"),
            Token::new(TokenType::Id, "d"),
            Token::new(TokenType::Rparen, ")"),
            Token::new(TokenType::Rparen, ")"),
        ];
        let mut parser = Parser::new(&tokens);
        fn id<'a>(parser: &mut Parser<'a, TokenType>) -> Result<Expr<'a>, ParserError<TokenType>> {
            let id = parser.expect(TokenType::Id)?;
            Ok(Expr::Id(id))
        }
        fn call<'a>(parser: &mut Parser<'a, TokenType>) -> Result<Expr<'a>, ParserError<TokenType>> {
            parser.expect(TokenType::Lparen)?;
            let callee = expr(parser)?;
            let args = parser.any(expr);
            parser.expect(TokenType::Rparen)?;
            Ok(Expr::Call(Box::new(callee), args))
        }
        fn expr<'a>(parser: &mut Parser<'a, TokenType>) -> Result<Expr<'a>, ParserError<TokenType>> {
            parser.either(id, call)
        }
        match expr(&mut parser) {
            Ok(tree) => assert_eq!(tree.to_string(), "a(b()(), c(d))"),
            Err(_) => panic!("tree didn't parse")
        }
    }
}
