import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.split(__file__)[0], '..')))

import typing
from parsey import Parser, String
class JsonParser:
    json_parser = None
    @staticmethod
    def get_parser() -> Parser:
        if JsonParser.json_parser is not None:
            return JsonParser.json_parser
        whitespace = String.char_from(' \t\n\r')
        comment = String.string('//').times(1) + Parser.condition_else(String.char_from('\n') | Parser.eof, Parser.any()).many() + (String.char_from('\n') | Parser.eof).times(1)
        ignores = (comment | whitespace).many()
        lexeme = lambda p: p << ignores

        # Punctuation
        lbrace = lexeme(String.string("{"))
        rbrace = lexeme(String.string("}"))
        lbrack = lexeme(String.string("["))
        rbrack = lexeme(String.string("]"))
        colon = lexeme(String.string(":"))
        comma = lexeme(String.string(","))

        # Primitives
        true = lexeme(String.string("true")).result(True)
        false = lexeme(String.string("false")).result(False)
        null = lexeme(String.string("null")).result(None)
        number_int = lexeme(String.regex(r"-?(0|[1-9][0-9]*)")).map(int)
        number_float = lexeme(String.regex(r"-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)")).map(float)
        number = (number_float | number_int)

        string_part_peek = String.char_from('"\\')
        string_part = Parser.condition_else(peek=string_part_peek, else_=Parser.any())
        string_esc = String.string("\\") >> (
            String.string("\\")
            | String.string("/")
            | String.string('"')
            | String.string("b").result("\b")
            | String.string("f").result("\f")
            | String.string("n").result("\n")
            | String.string("r").result("\r")
            | String.string("t").result("\t")
            | (
                String.char_from('u') >> String.char_from('0123456789abcdefABCDEF').times(min=4)
              ).map(
                    lambda x: chr(int(''.join(x), 16))
                )
            )
        quoted = lexeme(String.string('"') >> (string_part | string_esc).many().map(''.join) << String.string('"'))


        # Data structures
        json_value = Parser()
        object_pair = Parser.seq(quoted << colon, json_value).map(tuple)
        json_object = (lbrace >> object_pair.sep_by(comma, allowed_extra_sep=True).map(dict) << rbrace)
        array = lbrack >> json_value.sep_by(comma, allowed_extra_sep=True) << rbrack

        # Everything
        json_value.set_parser(quoted | number | json_object | array | true | false | null)
        json_doc = ignores >> json_value

        JsonParser.json_parser = json_doc
        return JsonParser.json_parser

    def parse_from_string(json_string: str) -> typing.Any:
        json_parser = JsonParser.get_parser()
        return json_parser.parse(json_string)
    
    @staticmethod
    def parse_from_file(file: str) -> typing.Any:
        with open(file, 'r') as input_stream:
            json_string = input_stream.read()
        return JsonParser.parse_from_string(json_string=json_string)

if __name__ == '__main__':
    def test():
        json_str = ''.join(
            (
                '{\n',
                '"int": 1,\n',
                '"string": "hello",//this is a comment\n\n',
                '//this is another comment\n',
                '"a list": [1, 2, 3, ["abc", 7]],\n',
                '"escapes": "\n \u24D2 中文",\n',
                '"nested": {"x": "y"},\n',
                '"other": [true, false, null],\n',
                '}//comment2122'
            )
        )   
        r = JsonParser.parse_from_string(json_str)
        print(r)
        assert (
            r
            == {
                "int": 1,
                "string": "hello",
                "a list": [1, 2, 3, ['abc', 7]],
                "escapes": "\n ⓒ 中文",
                "nested": {"x": "y"},
                "other": [True, False, None],
            }
        )
        # r = json_doc.parse(
        #         r"""
        # {
        #     "int": 1,
        #     "string": "hello",
        #     "a list": [1, 2, 3],
        #     "escapes": "\n \u24D2 中文",
        #     "nested": {"x": "y"},
        #     "other": [true, false, null]
        # }
        # """
        # )
        # print(r)

    test()