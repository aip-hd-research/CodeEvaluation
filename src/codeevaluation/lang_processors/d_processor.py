import typing as tp
from pathlib import Path

import tree_sitter as ts

from .tree_sitter_processor import TREE_SITTER_ROOT, TreeSitterLangProcessor

D_TOKEN2CHAR = {
    "STOKEN00": "//",
    "STOKEN01": "/*",
    "STOKEN02": "*/",
    "STOKEN03": "/**",
    "STOKEN04": "**/",
    "STOKEN06": "\\n",
    "STOKEN07": "\\r",
    "STOKEN08": ";",
    "STOKEN09": "{",
    "STOKEN10": "}",
    "STOKEN11": r"\'",
    "STOKEN12": r"\"",
    "STOKEN13": r"\\",
}
D_CHAR2TOKEN = {value: " " + key + " " for key, value in D_TOKEN2CHAR.items()}


class DProcessor(TreeSitterLangProcessor):
    TREESITTER_REPOSITORY = "gdamore/tree-sitter-d"

    def __init__(self, root_folder: Path = TREE_SITTER_ROOT) -> None:
        super().__init__(
            language="d",
            ast_nodes_type_string=[
                "comment",
                "string_literal",
            ],
            stokens_to_chars=D_TOKEN2CHAR,
            chars_to_stokens=D_CHAR2TOKEN,
            root_folder=root_folder,
        )

    def extract_functions(
        self, code: tp.Union[str, tp.List[str]], tokenized: bool = True
    ) -> tp.Tuple[tp.List[str], tp.List[str]]:
        if isinstance(code, list):
            code = " ".join(code)
        if tokenized:
            code = self.detokenize_code(code)
        if isinstance(code, str):
            code = bytes(code, "utf-8")
        ast = self.get_ast(code)

        class_funcs, standalone_funcs = self._get_functions_from_ast(
            code, ast.root_node
        )

        if tokenized:
            class_funcs = [
                " ".join(self.tokenize_code(f)) for f in class_funcs
            ]
            standalone_funcs = [
                " ".join(self.tokenize_code(f)) for f in standalone_funcs
            ]

        return standalone_funcs, class_funcs

    def _get_functions_from_ast(
        self, code: str, node: ts.Node
    ) -> tp.Tuple[tp.List[str], tp.List[str]]:
        class_funcs = []
        standalone_funcs = []

        if is_class_func(node):
            class_funcs.append(node.text.decode("utf-8"))
        elif is_standalone_func(node):
            standalone_funcs.append(node.text.decode("utf-8"))

        for child in node.children:
            (
                child_class_funcs,
                child_standalone_funcs,
            ) = self._get_functions_from_ast(code, child)
            class_funcs.extend(child_class_funcs)
            standalone_funcs.extend(child_standalone_funcs)

        return class_funcs, standalone_funcs

    def get_function_name(self, function):
        return self.get_first_non_bracket_token_before_first_parenthesis(
            function
        )


def is_class_func(node):
    return (
        is_func(node)
        and (has_struct_parent(node) or has_class_parent(node))
        and not is_static(node)
    )


def has_struct_parent(node):
    if node.parent is None:
        return False
    if node.parent.type == "struct_declaration":
        return True
    return has_struct_parent(node.parent)


def is_func(node):
    return node.type == "function_declaration" and not is_abstract(node)


def is_func_literal(node):
    return node.type == "function_literal"


def is_static(node):
    if node.prev_named_sibling is None:
        return False
    sibling = node.prev_named_sibling
    return sibling.type == "static"


def is_abstract(node):
    if node.prev_named_sibling is None:
        return False
    sibling = node.prev_named_sibling
    return sibling.type == "abstract"


def has_class_parent(node):
    if node.parent is None:
        return False
    if node.parent.type == "class_declaration":
        return True
    return has_class_parent(node.parent)


def has_body_longer_than(node, length):
    function_bodies = [
        child for child in node.children if child.type == "function_body"
    ]
    if len(function_bodies) == 0 or len(function_bodies) > 1:
        return False
    function_body = function_bodies[0]
    return (function_body.end_byte - function_body.start_byte) > length


def is_standalone_func(node):
    return (
        is_func(node)
        and (
            not (has_struct_parent(node) or has_class_parent(node))
            or is_static(node)
        )
    ) or is_func_literal(
        node
    )  # can never be part of a class or struct