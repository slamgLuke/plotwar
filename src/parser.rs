// parser.rs

// grammar:
// EXP = TERM | TERM (('+' | '-') TERM)*                // left associative
// TERM = POWER | POWER (('*' | '/') POWER)*            // left associative
// POWER = FACTOR | FACTOR ('^' FACTOR)*                // left associative
// FACTOR = '-' FACTOR | '|' EXP '|' | 'l' '(' EXP ')' | '(' EXP ')' | NUM | 'x'

// Tokens and Operators
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    Plus,
    Minus,
    Mul,
    Div,
    Pow,
    Abs,
    Exp,
    Ln,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    X,
    Op(Op),
    LParen,
    RParen,
}

impl Token {
    fn unwrap_op(&self) -> Op {
        match self {
            Token::Op(op) => op.clone(),
            _ => panic!("Expected operator"),
        }
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn from_chars(input: &[char]) -> Result<Self, String> {
        let tokens = tokenize(input)?;
        Ok(Parser { tokens, pos: 0 })
    }

    fn parse(&mut self) -> Result<Exp, String> {
        self.parse_exp()
    }

    fn parse_exp(&mut self) -> Result<Exp, String> {
        let mut left = self.parse_term()?;
        while let Some(Token::Op(Op::Plus)) | Some(Token::Op(Op::Minus)) = self.peek() {
            let op = BinOp::from_op(self.next().unwrap().unwrap_op());
            let right = self.parse_term()?;
            left = Exp::Term(Box::new(left), op, Box::new(right));
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Exp, String> {
        let mut left = self.parse_power()?;
        while let Some(Token::Op(Op::Mul)) | Some(Token::Op(Op::Div)) = self.peek() {
            let op = BinOp::from_op(self.next().unwrap().unwrap_op());
            let right = self.parse_power()?;
            left = Exp::Factor(Box::new(left), op, Box::new(right));
        }
        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Exp, String> {
        let mut left = self.parse_factor()?;
        while let Some(Token::Op(Op::Pow)) = self.peek() {
            self.next(); // Consume the operator
            let right = self.parse_factor()?;
            left = Exp::Factor(Box::new(left), BinOp::Pow, Box::new(right));
        }
        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Exp, String> {
        match self.next() {
            Some(Token::Op(Op::Minus)) => {
                let factor = self.parse_factor()?;
                Ok(Exp::UnExp(UnOp::Neg, Box::new(factor)))
            }
            Some(Token::Op(Op::Abs)) => {
                let exp = self.parse_exp()?;
                if self.next() != Some(Token::Op(Op::Abs)) {
                    return Err("Expected closing '|'".to_string());
                }
                Ok(Exp::UnExp(UnOp::Abs, Box::new(exp)))
            }
            Some(Token::Op(Op::Ln)) => {
                if self.next() != Some(Token::LParen) {
                    return Err("Expected '(' after 'ln'".to_string());
                }
                let exp = self.parse_exp()?;
                if self.next() != Some(Token::RParen) {
                    return Err("Expected closing ')' after 'ln'".to_string());
                }
                Ok(Exp::UnExp(UnOp::Ln, Box::new(exp)))
            }
            Some(Token::LParen) => {
                let exp = self.parse_exp()?;
                if self.next() != Some(Token::RParen) {
                    return Err("Expected closing ')'".to_string());
                }
                Ok(Exp::Paren(Box::new(exp)))
            }
            Some(Token::Num(num)) => Ok(Exp::Num(num)),
            Some(Token::X) => Ok(Exp::X),
            _ => Err("Unexpected token in factor".to_string()),
        }
    }

    fn parse_parenthesized(&mut self) -> Result<Exp, String> {
        let exp = self.parse_exp()?;
        if !matches!(self.next(), Some(Token::RParen)) {
            return Err("Expected closing ')'".to_string());
        }
        Ok(Exp::Paren(Box::new(exp)))
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }
}

// Tokenization
fn tokenize(input: &[char]) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut num = String::new();
    let mut is_num = false;

    for &c in input {
        if c.is_whitespace() {
            continue;
        }

        if is_num {
            if c.is_digit(10) || c == '.' {
                num.push(c);
                continue;
            } else {
                tokens.push(Token::Num(
                    num.parse().map_err(|_| "Invalid number".to_string())?,
                ));
                num.clear();
                is_num = false;
            }
        }

        match c {
            '+' => tokens.push(Token::Op(Op::Plus)),
            '-' => tokens.push(Token::Op(Op::Minus)),
            '*' => tokens.push(Token::Op(Op::Mul)),
            '/' => tokens.push(Token::Op(Op::Div)),
            '^' => tokens.push(Token::Op(Op::Pow)),
            '|' => tokens.push(Token::Op(Op::Abs)),
            '(' => tokens.push(Token::LParen),
            ')' => tokens.push(Token::RParen),
            'x' | 'X' => tokens.push(Token::X),
            'e' => tokens.push(Token::Num(std::f64::consts::E)),
            'p' => tokens.push(Token::Num(std::f64::consts::PI)),
            'l' => tokens.push(Token::Op(Op::Ln)),
            _ => {
                if c.is_digit(10) || c == '.' {
                    num.push(c);
                    is_num = true;
                } else {
                    return Err("Unexpected character".to_string());
                }
            }
        }
    }

    if !num.is_empty() {
        tokens.push(Token::Num(
            num.parse().map_err(|_| "Invalid number".to_string())?,
        ));
    }

    Ok(tokens)
}

// Expressions and Operators

#[derive(Debug, Clone, PartialEq)]
enum Exp {
    Term(Box<Exp>, BinOp, Box<Exp>),
    Factor(Box<Exp>, BinOp, Box<Exp>),
    UnExp(UnOp, Box<Exp>),
    Paren(Box<Exp>),
    Num(f64),
    X,
}

impl Exp {
    fn eval(&self, x: f64) -> f64 {
        match self {
            Exp::Term(left, op, right) => op.apply(left.eval(x), right.eval(x)),
            Exp::Factor(left, op, right) => op.apply(left.eval(x), right.eval(x)),
            Exp::UnExp(op, operand) => op.apply(operand.eval(x)),
            Exp::Paren(exp) => exp.eval(x),
            Exp::Num(num) => *num,
            Exp::X => x,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum UnOp {
    Neg,
    Abs,
    Ln,
}

impl BinOp {
    fn from_op(op: Op) -> Self {
        match op {
            Op::Plus => BinOp::Add,
            Op::Minus => BinOp::Sub,
            Op::Mul => BinOp::Mul,
            Op::Div => BinOp::Div,
            Op::Pow => BinOp::Pow,
            _ => panic!("Unexpected operator"),
        }
    }

    fn apply(&self, left: f64, right: f64) -> f64 {
        match self {
            BinOp::Add => left + right,
            BinOp::Sub => left - right,
            BinOp::Mul => left * right,
            BinOp::Div => left / right,
            BinOp::Pow => left.powf(right),
        }
    }
}

impl UnOp {
    fn from_op(op: Op) -> Self {
        match op {
            Op::Minus => UnOp::Neg,
            Op::Abs => UnOp::Abs,
            Op::Ln => UnOp::Ln,
            _ => panic!("Unexpected operator"),
        }
    }

    fn apply(&self, operand: f64) -> f64 {
        match self {
            UnOp::Neg => -operand,
            UnOp::Abs => operand.abs(),
            UnOp::Ln => operand.ln(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn into_input(input: &str) -> Vec<char> {
        input.chars().collect()
    }

    fn assert_approx_eq(exp: &Exp, x: f64, expected: f64, tol: f64) {
        let result = exp.eval(x);
        if (result - expected).abs() > tol {
            panic!(
            "Assertion failed: result = {}, expected = {}, difference = {} exceeds tolerance = {}",
            result,
            expected,
            (result - expected).abs(),
            tol
        );
        }
    }

    const TOL: f64 = 0.005;

    #[test]
    fn test1() {
        let input = into_input("-2+((0))-(-(x+2)) + 1/1.000 - (0.000 +1.0*(2.0*3/(2+1))) + 3");
        let mut parser = Parser::from_chars(&input).unwrap();
        let exp = parser.parse().unwrap();
        dbg!(exp.clone());
        assert_eq!(exp.eval(1.232), 3.232);
    }

    #[test]
    fn test2() {
        for x in -10..10 {
            let input = into_input("2 + x - 1 * 2 / 1 - 0.924 + 0.924");
            let mut parser = Parser::from_chars(&input).unwrap();
            let exp = parser.parse().unwrap();
            dbg!(exp.clone());
            assert_approx_eq(&exp, x as f64, x as f64, TOL);
        }
    }

    #[test]
    fn test3() {
        let input = into_input("l(x) /(241-2^2-1) +1/  x");
        let mut parser = Parser::from_chars(&input).unwrap();
        let exp = parser.parse().unwrap();
        dbg!(exp.clone());
        assert_approx_eq(&exp, 0.7027659726, 1.426, TOL);
    }

    #[test]
    fn test4() {
        let input = into_input("-x+1");
        let mut parser = Parser::from_chars(&input).unwrap();
        let exp = parser.parse().unwrap();
        dbg!(exp.clone());
        assert_approx_eq(&exp, 13.1667, -12.1667, TOL);
    }

    #[test]
    fn precedente_test() {
        let input = into_input("(-x - 1) - |2| + 3 * 4 ^ 5 - 6 / l(7) + 8 - 9 * 10");
        let mut parser = Parser::from_chars(&input).unwrap();
        let exp = parser.parse().unwrap();
        dbg!(exp.clone());
        assert_approx_eq(&exp, 2983.9166099, 0.0, TOL);

        let input = into_input("(2+3) * 4 - |2+3| * x");
        let mut parser = Parser::from_chars(&input).unwrap();
        let exp = parser.parse().unwrap();
        dbg!(exp.clone());
        assert_approx_eq(&exp, 4.0, 0.0, TOL);
    }
}
