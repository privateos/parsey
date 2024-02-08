from __future__ import annotations

import dataclasses
import typing
import functools

StreamType = typing.Union[str, bytes, list]
noop = lambda x: x

class ParseError(RuntimeError):
    def __init__(self, expected: typing.FrozenSet[str], stream: StreamType, index: int) -> None:
        self.expected = expected
        self.stream = stream
        self.index = index

    def line_info(self) -> str:
        try:
            return "{}:{}".format(*String.line_info_at(self.stream, self.index))
        except (TypeError, AttributeError):  # not a str
            return str(self.index)

    def __str__(self):
        expected_list = sorted(repr(e) for e in self.expected)

        if len(expected_list) == 1:
            return f"expected {expected_list[0]} at {self.line_info()}"
        else:
            return f"expected one of {', '.join(expected_list)} at {self.line_info()}"


@dataclasses.dataclass
class Result:
    status: bool
    index: int
    value: typing.Any
    furthest: int
    expected: typing.FrozenSet[str]

    @staticmethod
    def success(index: int, value: typing.Any):
        return Result(True, index, value, -1, frozenset())

    @staticmethod
    def failure(index: int, expected: str):
        return Result(False, -1, None, index, frozenset([expected]))

    # collect the furthest failure from self and other
    def aggregate(self, other: typing.Optional[Result]) -> Result:
        if not other:
            return self

        if self.furthest > other.furthest:
            return self
        elif self.furthest == other.furthest:
            # if we both have the same failure index, we combine the expected messages.
            return Result(self.status, self.index, self.value, self.furthest, self.expected | other.expected)
        else:
            return Result(self.status, self.index, self.value, other.furthest, other.expected)



class Parser:
    """
    A Parser is an object that wraps a function whose arguments are
    a string to be parsed and the index on which to begin parsing.
    The function should return either Result.success(next_index, value),
    where the next index is where to continue the parse and the value is
    the yielded value, or Result.failure(index, expected), where expected
    is a string indicating what was expected, and the index is the index
    of the failure.
    """
    # @staticmethod
    # def forward() -> Parser:
    #     pass
    # forward_parser: typing.Dict[str, ForwardParser] = {}
    # @staticmethod
    # def define(name: str, parser: Parser) -> None:
    #     fp = Parser.forward_parser.get(name)
    #     if fp is None:
    #         # fp = ForwardParser()
    #         Parser.forward_parser[name] = fp

    #     fp.set_parser(parser)


    # @staticmethod
    # def get(name: str) -> typing.Optional[Parser]:
        

    @staticmethod
    def peek(parser: Parser) -> Parser:
        """
        Returns a lookahead parser that parses the input stream without consuming
        chars.
        """

        @Parser
        def peek_parser(stream: StreamType, index: int) -> Result:
            result = parser(stream, index)
            if result.status:
                return Result.success(index, result.value)
            else:
                return result

        return peek_parser

    @staticmethod
    def success(value: typing.Any) -> Parser:
        """
        Returns a parser that does not consume any of the stream, but
        produces ``value``.
        """
        return Parser(lambda _, index: Result.success(index, value))

    @staticmethod
    def fail(expected: str) -> Parser:
        """
        Returns a parser that always fails with the provided error message.
        """
        return Parser(lambda _, index: Result.failure(index, expected))

    @staticmethod
    def alt(*parsers: Parser) -> Parser:
        """
        Creates a parser from the passed in argument list of alternative
        parsers, which are tried in order, moving to the next one if the
        current one fails.
        """
        if not parsers:
            return Parser.fail("<empty alt>")

        @Parser
        def alt_parser(stream, index):
            result = None
            for parser in parsers:
                result = parser(stream, index).aggregate(result)
                if result.status:
                    return result

            return result

        return alt_parser

    @staticmethod
    def seq(*parsers: Parser) -> Parser:
        """
        Takes a list of parsers, runs them in order,
        and collects their individuals results in a list,
        or in a dictionary if you pass them as keyword arguments.
        """
        if not parsers:
            return Parser.success([])

        @Parser
        def seq_parser(stream, index):
            result = None
            values = []
            for parser in parsers:
                result = parser(stream, index).aggregate(result)
                if not result.status:
                    return result
                index = result.index
                values.append(result.value)
            return Result.success(index, values).aggregate(result)

        return seq_parser

    @staticmethod
    def generate(fn: typing.Generator) -> Parser:
        """
        Creates a parser from a generator function
        """
        if isinstance(fn, str):
            return lambda f: Parser.generate(f).desc(fn)


        @Parser
        @functools.wraps(fn)
        def generated(stream: StreamType, index: int) -> Result:
            # start up the generator
            iterator = fn()

            result = None
            value = None
            try:
                while True:
                    next_parser = iterator.send(value)
                    result = next_parser(stream, index).aggregate(result)
                    if not result.status:
                        return result
                    value = result.value
                    index = result.index
            except StopIteration as stop:
                returnVal = stop.value
                if isinstance(returnVal, Parser):
                    return returnVal(stream, index).aggregate(result)

                return Result.success(index, returnVal).aggregate(result)

        return generated

    @staticmethod
    def condition(
        peek: Parser, 
        if_: Parser, 
        else_: Parser
    ) -> Parser:
        @Parser
        def fn(stream: StreamType, index: int) -> Result:
            result = peek(stream=stream, index=index)
            if result.status:
                return if_(stream=stream, index=index)
            else:
                return else_(stream=stream, index=index)
        return fn
                 
    @staticmethod
    def condition_if(peek: Parser, if_: Parser) -> Parser:
        
        @Parser
        def fn(stream: StreamType, index: int) -> Result:
            result = peek(stream=stream, index=index)
            if result.status:
                return if_(stream=stream, index=index)
            return Result.failure(index=index, expected=f'condition_if={peek}')
        
        return fn
    
    @staticmethod
    def condition_else(peek: Parser, else_: Parser) -> Parser:
        
        @Parser
        def fn(stream: StreamType, index: int) -> Result:
            result = peek(stream=stream, index=index)
            if not result.status:
                return else_(stream=stream, index=index)
            return Result.failure(index=index, expected=f'condition_else={peek}')
        return fn

    @staticmethod
    def any() -> Parser:
        """
        Returns a parser that fails when the initial parser succeeds, and succeeds
        when the initial parser fails (consuming no input). A description must
        be passed which is used in parse failure messages.

        This is essentially a negative lookahead
        """
        @Parser
        def any_parser(stream: StreamType, index: int) -> Result:
            res = stream[index]
            return Result.success(index + 1, res)

        return any_parser

    def __init__(self, wrapped_fn: typing.Optional[typing.Callable[[StreamType, int], Result]] = None) -> None:
        """
        Creates a new Parser from a function that takes a stream
        and returns a Result.
        """
        # self.wrapped_fn = wrapped_fn
        self.wrapped_fn = None
        self.set_function(wrapped_fn)

    def __call__(self, stream: StreamType, index: int):
        return self.wrapped_fn(stream, index)

    def set_parser(self, parser: Parser) -> None:
        if self.wrapped_fn is not None:
            raise ValueError(f'wrapped_fn is not None')
        def fn(stream: StreamType, index: int) -> Result:
            return parser(stream, index)
        # self.wrapped_fn = fn
        self.set_function(function=fn)

    def set_function(self, function: typing.Callable[[StreamType, int], Result]) -> None:
        if self.wrapped_fn is not None:
            raise ValueError(f'wrapped_fn is not None')
        self.wrapped_fn = function

    def parse(self, stream: StreamType) -> typing.Any:
        """Parses a string or list of tokens and returns the result or raise a ParseError."""
        # (result, _) = (self << Parser.eof).parse_partial(stream)
        (result, _) = self.skip(Parser.eof).parse_partial(stream)
        return result

    def parse_partial(self, stream: StreamType) -> typing.Tuple[typing.Any, StreamType]:
        """
        Parses the longest possible prefix of a given string.
        Returns a tuple of the result and the unparsed remainder,
        or raises ParseError
        """
        result = self(stream, 0)

        if result.status:
            return (result.value, stream[result.index :])
        else:
            raise ParseError(result.expected, stream, result.furthest)

    #bind的作用和map相同，但是bind_fn会返回一个Parser
    def bind(self, bind_fn: typing.Callable[[typing.Any], Parser]) -> Parser:
        @Parser
        def bound_parser(stream: StreamType, index: int) -> Result:
            result = self(stream, index)

            if result.status:
                next_parser = bind_fn(result.value)
                return next_parser(stream, result.index).aggregate(result)
            else:
                return result

        return bound_parser

    #map把之前解析出来的结果当作一个整体传入到函数中
    def map(self, map_function: typing.Callable[[typing.Any], typing.Any]) -> Parser:
        # """
        # Returns a parser that transforms the produced value of the initial parser with map_function.
        # """
        # return self.bind( lambda res: Parser.success(map_function(res)) )
        @Parser
        def map_parser(stream: StreamType, index: int) -> Result:
            result = self(stream, index)
            if result.status:
                result.value = map_function(result.value)
            
            return result
        return map_parser
        
    #combine假设之前解析出来的结果是list,把list unpack作为函数的输入
    def combine(self, combine_fn: typing.Callable[[typing.Any], typing.Any]) -> Parser:
        """
        Returns a parser that transforms the produced values of the initial parser
        with ``combine_fn``, passing the arguments using ``*args`` syntax.

        The initial parser should return a list/sequence of parse results.
        """
        # return self.bind(lambda res: success(combine_fn(*res)))
        return self.map(lambda res: combine_fn(*res))


    def combine_dict(self, combine_fn: typing.Callable[[typing.Any], typing.Any]) -> Parser:
        """
        Returns a parser that transforms the value produced by the initial parser
        using the supplied function/callable, passing the arguments using the
        ``**kwargs`` syntax.

        The value produced by the initial parser must be a mapping/dictionary from
        names to values, or a list of two-tuples, or something else that can be
        passed to the ``dict`` constructor.

        If ``None`` is present as a key in the dictionary it will be removed
        before passing to ``fn``, as will all keys starting with ``_``.
        """
        return self.bind(
            lambda res: Parser.success(
                combine_fn(
                    **{
                        k: v
                        for k, v in dict(res).items()
                        if k is not None and not (isinstance(k, str) and k.startswith("_"))
                    }
                )
            )
        )
        # return self.map(
        #     lambda res:combine_fn(**{for k: v in dict(res).items()})
        # )


    def then(self, other: Parser) -> Parser:
        """
        Returns a parser which, if the initial parser succeeds, will
        continue parsing with ``other``. This will produce the
        value produced by ``other``.

        """
        return Parser.seq(self, other).combine(lambda left, right: right)
        # return Parser.seq(self, other).map(lambda left, right: right)

    def skip(self, other: Parser) -> Parser:
        """
        Returns a parser which, if the initial parser succeeds, will
        continue parsing with ``other``. It will produce the
        value produced by the initial parser.
        """
        # return seq(self, other).combine(lambda left, right: left)
        return Parser.seq(self, other).combine(lambda left, right: left)

    def result(self, value: typing.Any) -> Parser:
        """
        Returns a parser that, if the initial parser succeeds, always produces
        the passed in ``value``.
        """
        # return self >> success(value)
        return self.map(lambda x: value)



    def many(self) -> Parser:
        """
        Returns a parser that expects the initial parser 0 or more times, and
        produces a list of the results.
        """
        return self.times(0, float("inf"))

    def times(self, min: int, max: typing.Optional[int] = None) -> Parser:
        """
        Returns a parser that expects the initial parser at least ``min`` times,
        and at most ``max`` times, and produces a list of the results. If only one
        argument is given, the parser is expected exactly that number of times.
        """
        if max is None:
            max = min

        @Parser
        def times_parser(stream: StreamType, index: int) -> Result:
            values = []
            times = 0
            result = None

            while times < max:
                result = self(stream, index).aggregate(result)
                if result.status:
                    values.append(result.value)
                    index = result.index
                    times += 1
                elif times >= min:
                    break
                else:
                    return result

            return Result.success(index, values).aggregate(result)

        return times_parser

    def at_most(self, n: int) -> Parser:
        """
        Returns a parser that expects the initial parser at most ``n`` times, and
        produces a list of the results.
        """
        return self.times(0, n)

    def at_least(self, n: int) -> Parser:
        """
        Returns a parser that expects the initial parser at least ``n`` times, and
        produces a list of the results.
        """
        return self.times(n) + self.many()
        #返回的两个列表相加
        # return Parser.seq(self.times(n), self.many()).map(lambda x, y: x + y)

    def optional(self, default: typing.Any = None) -> Parser:
        """
        Returns a parser that expects the initial parser zero or once, and maps
        the result to a given default value in the case of no match. If no default
        value is given, ``None`` is used.
        """
        return self.times(0, 1).map(lambda v: v[0] if v else default)

    def until(self, 
            other: Parser, 
            min: int = 0, 
            max: int = float("inf"), 
            consume_other: bool = False
    ) -> Parser:
        """
        Returns a parser that expects the initial parser followed by ``other``.
        The initial parser is expected at least ``min`` times and at most ``max`` times.
        By default, it does not consume ``other`` and it produces a list of the
        results excluding ``other``. If ``consume_other`` is ``True`` then
        ``other`` is consumed and its result is included in the list of results.
        """

        @Parser
        def until_parser(stream: StreamType, index: int) -> Result:
            values = []
            times = 0
            while True:

                # try parser first
                res = other(stream, index)
                if res.status and times >= min:
                    if consume_other:
                        # consume other
                        values.append(res.value)
                        index = res.index
                    return Result.success(index, values)

                # exceeded max?
                if times >= max:
                    # return failure, it matched parser more than max times
                    return Result.failure(index, f"at most {max} items")

                # failed, try parser
                result = self(stream, index)
                if result.status:
                    # consume
                    values.append(result.value)
                    index = result.index
                    times += 1
                elif times >= min:
                    # return failure, parser is not followed by other
                    return Result.failure(index, "did not find other parser")
                else:
                    # return failure, it did not match parser at least min times
                    return Result.failure(index, f"at least {min} items; got {times} item(s)")

        return until_parser

    def sep_by(self, sep: Parser, *, min: int = 0, max: int = float("inf"), allowed_extra_sep: bool = False) -> Parser:
        """
        Returns a new parser that repeats the initial parser and
        collects the results in a list. Between each item, the ``sep`` parser
        is run (and its return value is discarded). By default it
        repeats with no limit, but minimum and maximum values can be supplied.
        """
        zero_times = Parser.success([])
        if max == 0:
            return zero_times
        if allowed_extra_sep:
            res = self.times(1) + ( (sep >> self).times(min - 1, max - 1) << sep.optional() )
        else:
            res = self.times(1) + (sep >> self).times(min - 1, max - 1)
        if min == 0:
            res |= zero_times
        return res

    def desc(self, description: str) -> Parser:
        """
        Returns a new parser with a description added, which is used in the error message
        if parsing fails.
        """

        @Parser
        def desc_parser(stream, index):
            result = self(stream, index)
            if result.status:
                return result
            else:
                return Result.failure(index, description)

        return desc_parser

    def mark(self) -> Parser:
        """
        Returns a parser that wraps the initial parser's result in a value
        containing column and line information of the match, as well as the
        original value. The new value is a 3-tuple:

        ((start_row, start_column),
         original_value,
         (end_row, end_column))
        """

        # @Parser.generate
        # def marked():
        #     start = yield String.line_info
        #     body = yield self
        #     end = yield String.line_info
        #     return (start, body, end)
        @Parser
        def marked(stream: StreamType, index: int) -> typing.Tuple:
            start = String.line_info_at(stream, index)
            body = self(stream, index)
            end = String.line_info_at(stream=body.index)
            return (start, body, end)

        return marked

    def tag(self, name: str) -> Parser:
        """
        Returns a parser that wraps the produced value of the initial parser in a
        2 tuple containing ``(name, value)``. This provides a very simple way to
        label parsed components
        """
        return self.map(lambda v: (name, v))

    def should_fail(self, description: str) -> Parser:
        """
        Returns a parser that fails when the initial parser succeeds, and succeeds
        when the initial parser fails (consuming no input). A description must
        be passed which is used in parse failure messages.

        This is essentially a negative lookahead
        """

        @Parser
        def fail_parser(stream: StreamType, index: int) -> Result:
            res = self(stream, index)
            if res.status:
                return Result.failure(index, description)
            return Result.success(index, res)

        return fail_parser
    

    ####################################################################
    def __add__(self, other: Parser) -> Parser:
        return Parser.seq(self, other).combine(lambda x, y: x + y)

    # def __mul__(self, other: typing.Union[int, typing.Type[range]]) -> Parser:
    #     if isinstance(other, range):
    #         return self.times(other.start, other.stop - 1)
    #     return self.times(other)

    def __or__(self, other: Parser) -> Parser:
        return Parser.alt(self, other)

    # haskelley operators, for fun #

    # >>
    def __rshift__(self, other: Parser) -> Parser:
        return self.then(other)

    # # <<
    def __lshift__(self, other: Parser) -> Parser:
        return self.skip(other)

    # def concat(self) -> Parser:
    #     """
    #     Returns a parser that concatenates together (as a string) the previously
    #     produced values.
    #     """
    #     return self.map("".join)

Parser.eof = Parser(lambda stream, index: Result.success(index, None) if index >= len(stream) else Result.failure(index, "EOF"))
Parser.index = Parser(lambda _, index: Result.success(index, index))

class String:
    @staticmethod
    def string(expected_string: str, transform: typing.Callable[[str], str] = noop) -> Parser:
        """
        Returns a parser that expects the ``expected_string`` and produces
        that string value.

        Optionally, a transform function can be passed, which will be used on both
        the expected string and tested string.
        """

        slen = len(expected_string)
        transformed_s = transform(expected_string)

        @Parser
        def string_parser(stream, index) -> Result:
            if transform(stream[index : index + slen]) == transformed_s:
                return Result.success(index + slen, expected_string)
            else:
                return Result.failure(index, expected_string)

        return string_parser

    @staticmethod
    def regex(exp: str, flags=0, group: int | str | tuple = 0) -> Parser:
        import re
        """
        Returns a parser that expects the given ``exp``, and produces the
        matched string. ``exp`` can be a compiled regular expression, or a
        string which will be compiled with the given ``flags``.

        Optionally, accepts ``group``, which is passed to re.Match.group
        https://docs.python.org/3/library/re.html#re.Match.group> to
        return the text from a capturing group in the regex instead of the
        entire match.
        """

        if isinstance(exp, (str, bytes)):
            exp = re.compile(exp, flags)
        if isinstance(group, (str, int)):
            group = (group,)

        @Parser
        def regex_parser(stream, index):
            match = exp.match(stream, index)
            if match:
                return Result.success(match.end(), match.group(*group))
            else:
                return Result.failure(index, exp.pattern)

        return regex_parser

    @staticmethod
    def test_item(func: typing.Callable[..., bool], description: str) -> Parser:
        """
        Returns a parser that tests a single item from the list of items being
        consumed, using the callable ``func``. If ``func`` returns ``True``, the
        parse succeeds, otherwise the parse fails with the description
        ``description``.
        """

        @Parser
        def test_item_parser(stream: StreamType, index: int) -> Result:
            if index < len(stream):
                if isinstance(stream, bytes):
                    # Subscripting bytes with `[index]` instead of
                    # `[index:index + 1]` returns an int
                    item = stream[index : index + 1]
                else:
                    item = stream[index]
                if func(item):
                    return Result.success(index + 1, item)
            return Result.failure(index, description)

        return test_item_parser

    @staticmethod
    def test_char(func: typing.Callable[..., bool], description: str) -> Parser:
        """
        Returns a parser that tests a single character with the callable
        ``func``. If ``func`` returns ``True``, the parse succeeds, otherwise
        the parse fails with the description ``description``.
        """
        # Implementation is identical to test_item
        return String.test_item(func, description)

    @staticmethod
    def match_item(
        item: typing.Any, 
        description: str = None, 
        eq: typing.Optional[typing.Callable[[typing.Any], bool]] = None
    ) -> Parser:
        """
        Returns a parser that tests the next item (or character) from the stream (or
        string) for equality against the provided item. Optionally a string
        description can be passed.
        """

        if description is None:
            description = str(item)

        if eq is None:
            fn = lambda i: item == i
        else:
            fn = eq
        return String.test_item(fn, description)

    @staticmethod
    def string_from(*strings: str, transform: typing.Callable[[str], str] = noop) -> Parser:
        """
        Accepts a sequence of strings as positional arguments, and returns a parser
        that matches and returns one string from the list. The list is first sorted
        in descending length order, so that overlapping strings are handled correctly
        by checking the longest one first.
        """
        # Sort longest first, so that overlapping options work correctly
        return Parser.alt(*(String.string(s, transform) for s in sorted(strings, key=len, reverse=True)))

    @staticmethod
    def char_from(string: str | bytes) -> Parser:
        """
        Accepts a string and returns a parser that matches and returns one character
        from the string.
        """
        if isinstance(string, bytes):
            return String.test_char(lambda c: c in string, b"[" + string + b"]")
        else:
            return String.test_char(lambda c: c in string, "[" + string + "]")

    @staticmethod
    def line_info_at(stream: StreamType, index: int) -> typing.Tuple[int, int]:
        if index > len(stream):
            raise ValueError("invalid index")
        line = stream.count("\n", 0, index)
        last_nl = stream.rfind("\n", 0, index)
        col = index - (last_nl + 1)
        return (line, col)

String.line_info = Parser(lambda stream, index: Result.success(index, String.line_info_at(stream, index)))

String.any_char = String.test_char(lambda c: True, "any character")

String.whitespace = String.regex(r"\s+")

String.letter = String.test_char(lambda c: c.isalpha(), "a letter")

String.digit = String.test_char(lambda c: c.isdigit(), "a digit")

String.decimal_digit = String.char_from("0123456789")
