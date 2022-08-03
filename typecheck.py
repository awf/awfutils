import types
import ast
import inspect
import astpretty

from textwrap import dedent


def get_ast_for_function(f):
    """
    Get AST for function f.

    This needs to do various fiddling with source lines
    """

    def normalize_source_lines(sourcelines):
        # Copied from pytorch:torch/jit/frontend.py
        """
        This helper function accepts a list of source lines. It finds the
        indentation level of the function definition (`def`), then it indents
        all lines in the function body to a point at or greater than that
        level. This allows for comments and continued string literals that
        are at a lower indentation than the rest of the code.
        Args:
            sourcelines: function source code, separated into lines by
                            the '\n' character
        Returns:
            A list of source lines that have been correctly aligned
        """

        def remove_prefix(text, prefix):
            return text[text.startswith(prefix) and len(prefix) :]

        # Find the line and line number containing the function definition
        for i, l in enumerate(sourcelines):
            if l.lstrip().startswith("def"):
                idx = i
                break
        fn_def = sourcelines[idx]

        # Get a string representing the amount of leading whitespace
        whitespace = fn_def.split("def")[0]

        # Add this leading whitespace to all lines before and after the `def`
        aligned_prefix = [
            whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]
        ]
        aligned_suffix = [
            whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1 :]
        ]

        # Put it together again
        aligned_prefix.append(fn_def)
        return aligned_prefix + aligned_suffix

    try:
        filename = inspect.getsourcefile(f)
        sourcelines, file_lineno = inspect.getsourcelines(f)
    except Exception as e:
        print("Could not get source for ", f)
        e.with_traceback("Exception")
        return None

    sourcelines = normalize_source_lines(sourcelines)
    source = "".join(sourcelines)
    dedent_src = dedent(source)

    return ast.parse(dedent_src, filename=filename)


class TypeCheckVisitor(ast.NodeTransformer):
    def visit_AnnAssign(self, node):
        # An assignment with a type annotation.
        # node.target : single node and can be a Name, a Attribute or a Subscript.
        # node.annotation : annotation, such as a Str or Name node.
        # node.value : single optional node.
        # node.simple : True for a Name node in target that do not appear between
        #               parentheses and are hence pure names and not expressions.

        if not node.simple:
            return node

        assert isinstance(node.target, ast.Name)  # Should be guaranteed by node.simple

        annot_str = ast.unparse(node.annotation)

        node_assert = ast.Assert(
            test=ast.Call(
                ast.Name("isinstance", ctx=ast.Load()),
                [
                    ast.Name(
                        node.target.id, ctx=ast.Load()
                    ),  # Convert ctx from Store to Load
                    node.annotation,
                ],
                [],
            ),
            msg=ast.Constant(
                value=f"{node.target.id} not of type {annot_str}", kind=None
            ),
        )
        node_assert = ast.copy_location(node_assert, node)
        ast.fix_missing_locations(node_assert)
        return [node, node_assert]


def typecheck(f, show_src=False):
    """
    Decorator which turns annotated assignments of the form
      x : T = e
    into
      x : T = e
      assert isinstance(x, T), "x is not of type T"

    EXAMPLE:

      def foo(x : int, y : float):
        z : int = x * y # This should error
        w : float = z * 3.2
        return w

      @typecheck
      def foo(x : int, y : float):
        z : int = x * y # Now it does
        w : float = z * 3.2
        return w

    OPERATION:

    This works by AST transformation, replacing the function foo above
    with the function

      def foo(x : int, y : float):
        z : int = x * y
        assert isinstance(z, int)
        w : float = z * 3.2
        assert isinstance(w, float)
        return w

    If you want to actually see the transformed code, call with show_src=True

      @functools.partial(typecheck, show_src=True)
      def foo(x : int, y : float):
        z : int = x * y # Now it does
        w : float = z * 3.2
        return w

    """

    # TODO:
    #   if not isinstance(x, T): raise TypeError("x is not of type T")

    node = get_ast_for_function(f)
    new_node = TypeCheckVisitor().visit(node)

    if show_src:
        print("typecheck: Transformed source code")
        new_src = ast.unparse(new_node)
        print(new_src)

    # Compile new AST to get wrapped function
    try:
        new_code = compile(new_node, filename="<typecheck>", mode="exec")
    except Exception as e:
        # Most compile errors are pretty opaque (https://stackoverflow.com/a/25795966)
        # So call astpretty.  If it succeeds, it's helpful to debug, if it fails, its
        # error messages are much more helpful
        msg = astpretty.pformat(new_node)
        print(msg)
        raise ValueError("See AST printed above") from e

    fns = tuple(k for k in new_code.co_consts if isinstance(k, types.CodeType))
    assert len(fns) == 1, "TODO: Will need a better way to find the code"

    f_code = fns[0]
    f_name = f.__name__ + "_typecheck_wrap"
    f_checked = types.FunctionType(f_code, globals=f.__globals__, name=f_name)
    f_checked.__wrapped__ = f
    return f_checked
