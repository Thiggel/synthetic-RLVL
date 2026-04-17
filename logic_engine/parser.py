import re
from fractions import Fraction

from .prover import *


class ParsingError(Exception):
    pass


class Symbols:
    symbols = {
        'not': '¬',
        '~': '¬',
        '∼': '¬',
        '−': '¬',
        'and': '∧',
        '^': '∧',
        '&': '∧',
        '.': '∧',
        '·': '∧',
        'or': '∨',
        'iff': '↔',
        '≡': '↔',
        '<->': '↔',
        'implies': '→',
        '⇒': '→',
        '⊃': '→',
        '->': '→',
        '>': '→',
        'forall': '∀',
        '⋀': '∀',
        'exists': '∃',
        '⋁': '∃',
        'falsum': '⊥',
        'XX': '⊥',
        '#': '⊥',
        'box': '□',
        '[]': '□',
        'dia': '♢',
        '<>': '♢',
    }

    word_like_keys = {'not', 'and', 'or', 'iff', 'implies', 'forall', 'exists', 'falsum', 'box', 'dia'}
    keys = sorted(symbols, key=len, reverse=True)
    patterns = []
    for key in keys:
        if key in word_like_keys:
            patterns.append(rf'\b{re.escape(key)}\b')
        else:
            patterns.append(re.escape(key))
    patterns.append(r'(?<![A-Za-z0-9_])A(?=[s-z](?:[A-Z(¬∀∃□♢]|$))')  # Forall shorthand: Ax(...)
    patterns.append(r'(?<![A-Za-z0-9_])E(?=[s-z](?:[A-Z(¬∀∃□♢]|$))')  # Exists shorthand: Ex(...)
    pattern = '|'.join(patterns)
    regex = re.compile(pattern)

    @classmethod
    def sub(cls, s):
        def repl(m):
            match = m.group(0)
            if match == 'A':
                return '∀'
            if match == 'E':
                return '∃'
            return cls.symbols[match]
        return cls.regex.sub(repl, s)


def split_line(line):
    parts = [s.strip() for s in re.split(r'[;|]', line)]
    if len(parts) != 2:
        raise ParsingError('Must provide justification separated by ";" or "|".')
    return parts


def strip_parens(s):
    while s and s[0] == '(' and s[-1] == ')':
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return s
        s = s[1:-1]
    return s


def find_main_connective(s, symbol):
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ')':
            depth += 1
        elif s[i] == '(':
            depth -= 1
        elif depth == 0 and s[i] == symbol:
            return i
    return -1


TERM_TOKEN_RE = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9-]*|\d+/\d+|\d+|[()+\-*/,])")


def tokenize_term(s):
    tokens = []
    idx = 0
    length = len(s)
    while idx < length:
        match = TERM_TOKEN_RE.match(s, idx)
        if not match:
            raise ParsingError(f'Invalid term syntax near "{s[idx:]}".')
        token = match.group(1)
        tokens.append(token)
        idx = match.end()
    tokens.append('EOF')
    return tokens


def parse_numeral(token):
    if '/' in token:
        numerator, denominator = token.split('/')
        return Numeral(Fraction(int(numerator), int(denominator)))
    return Numeral(Fraction(int(token), 1))


class TermParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def consume(self, expected=None):
        token = self.tokens[self.pos]
        if expected and token != expected:
            raise ParsingError(f'Expected "{expected}" but found "{token}".')
        self.pos += 1
        return token

    def parse(self):
        term = self.parse_sum()
        if self.peek() != 'EOF':
            raise ParsingError('Unexpected token in term.')
        return term

    def parse_sum(self):
        term = self.parse_product()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_product()
            term = Func(op, (term, right))
        return term

    def parse_product(self):
        term = self.parse_unary()
        while self.peek() in ('*', '/'):
            op = self.consume()
            right = self.parse_unary()
            term = Func(op, (term, right))
        return term

    def parse_unary(self):
        if self.peek() == '-':
            self.consume('-')
            expr = self.parse_unary()
            if isinstance(expr, Numeral):
                return Numeral(-expr.value)
            return Func('-', (expr,))
        return self.parse_atom()

    def parse_atom(self):
        token = self.consume()
        if token == '(':
            expr = self.parse_sum()
            self.consume(')')
            return expr
        if token.isdigit() or '/' in token:
            return parse_numeral(token)
        ident = token
        if len(ident) == 1 and ident in Const.names:
            return Const(ident)
        if len(ident) == 1 and ident in Var.names:
            return Var(ident)
        if self.peek() == '(':
            self.consume('(')
            args = []
            if self.peek() != ')':
                while True:
                    args.append(self.parse_sum())
                    if self.peek() != ',':
                        break
                    self.consume(',')
            self.consume(')')
            return Func(ident, tuple(args))
        return Const(ident)


def parse_term_expr(s):
    tokens = tokenize_term(s)
    parser = TermParser(tokens)
    return parser.parse()


def split_term_args(s):
    if not s:
        return []
    depth = 0
    start = 0
    args = []
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(s[start:i])
            start = i + 1
    args.append(s[start:])
    return [arg for arg in (a.strip() for a in args) if arg]


def _parse_formula(f):
    f = strip_parens(f)
    
    # Falsum
    if f == '⊥':
        return Falsum()
    
    # Prop vars
    m = re.fullmatch(r'[A-Z]', f)
    if m:
        return PropVar(f)
    
    # Equality
    idx = find_main_connective(f, '=')
    if idx != -1:
        left = parse_term_expr(f[:idx])
        right = parse_term_expr(f[idx + 1:])
        return Eq(left, right)

    # Binary connectives
    connectives = [('↔', Iff), ('→', Imp), ('∨', Or), ('∧', And)]

    for sym, cls in connectives:
        idx = find_main_connective(f, sym)
        if idx == -1:
            continue
        left = _parse_formula(f[:idx])
        right = _parse_formula(f[idx + 1:])
        return cls(left, right)
    
    # Negation
    if f.startswith('¬'):
        return Not(_parse_formula(f[1:]))
    
    # Quantifiers
    m = re.match(r'(∀|∃)([A-Za-z_][A-Za-z_0-9-]*)', f)
    if m:
        var = Var(m.group(2))
        inner = _parse_formula(f[len(m.group(0)):])
        if m.group(1) == '∀':
            return Forall(var, inner)
        return Exists(var, inner)

    # Predicates with explicit argument lists
    m = re.fullmatch(r'([A-Z][A-Za-z0-9_]*)\((.*)\)', f)
    if m:
        name, arg_str = m.groups()
        args = tuple(parse_term_expr(arg) for arg in split_term_args(arg_str))
        return Pred(name, args)

    # Legacy predicates
    m = re.fullmatch(r'([A-Z])([a-z]+)', f)
    if m:
        args = tuple(Const(t) if t in Const.names else Var(t) for t in m.group(2))
        return Pred(m.group(1), args)

    # Modal operators
    if f.startswith('□'):
        return Box(_parse_formula(f[1:]))
    if f.startswith('♢'):
        return Dia(_parse_formula(f[1:]))

    raise ParsingError(f'Could not parse formula: "{f}".')


def parse_formula(f):
    f = ''.join(Symbols.sub(f).split())
    return _parse_formula(f)


def parse_assumption(a):
    a = ''.join(Symbols.sub(a).split())
    if a == '□':
        return BoxMarker()
    return _parse_formula(a)


RULE_ALIASES = {
    'AI': '∀I',
    'AE': '∀E',
    'EI': '∃I',
    'EE': '∃E',
    'CI': '∧I',
    'CE': '∧E',
    'VI': '∨I',
    'VE': '∨E',
    'II': '→I',
    'IE': '→E',
    'BI': '↔I',
    'BE': '↔E',
    'NI': '¬I',
    'NE': '¬E',
    'ALG': 'ALG',
    'ARITH': 'ARITH',
    'CANCEL': 'CANCEL',
    'FACT': 'FACT',
}


def parse_rule(rule):
    rule = ''.join(rule.split())
    rule = RULE_ALIASES.get(rule, rule)
    if not (rule.isascii() and rule.isalpha() and rule.isupper()):
        rule = Symbols.sub(rule)
    for r in Rules.rules:
        if r.name == rule:
            return r
    raise ParsingError(f'Could not parse rule of inference: "{rule}".')


def parse_citations(citations):
    citations = ''.join(citations.split())

    c_list = []
    for c in citations.split(','):
        m = re.fullmatch(r'(\d+)-(\d+)', c)
        if m:
            pair = (int(m.group(1)), int(m.group(2)))
            c_list.append(pair)
            continue
        try:
            c_list.append(int(c))
        except ValueError:
            raise ParsingError(f'Could not parse citations: "{citations}".')
    return tuple(c_list)


def parse_justification(j):
    parts = j.split(',', maxsplit=1)
    r = parse_rule(parts[0])
    if len(parts) == 1:
        return Justification(r, ())
    c = parse_citations(parts[1])
    return Justification(r, c)


def parse_line(line):
    f, j = split_line(line)
    return parse_formula(f), parse_justification(j)
