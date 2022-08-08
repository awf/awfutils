import types
import ast
import inspect
import astpretty

from textwrap import dedent

from icecream import ic  # while debugging


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

    return ast.parse(dedent_src, filename=filename), filename


class TypeCheckVisitor(ast.NodeTransformer):
    def make_assert_node(self, name: str, annotation: ast.expr):
        annot_str = ast.unparse(annotation)

        # typecheck.assert_is_instance
        assert_is_instance = ast.Attribute(
            value=ast.Name("typecheck", ctx=ast.Load()),
            attr="assert_is_instance",
            ctx=ast.Load(),
        )

        node = ast.Expr(
            ast.Call(
                assert_is_instance,
                [
                    ast.Name(name, ctx=ast.Load()),
                    annotation,
                    ast.Constant(value=name, kind=None),
                    ast.Constant(value=annot_str, kind=None),
                ],
                [],
            )
        )

        # Assertion is associated with source location of the annotation - normally
        # on the same line, but if not, still arguably correct
        node = ast.copy_location(node, annotation)
        ast.fix_missing_locations(node)

        return node

    def visit_FunctionDef(self, node):
        """
        Add asserts for args
        """
        # node.name : raw string of the function name.
        # node.args : arguments node.
        # node.body : list of nodes inside the function.
        # node.decorator_list : list of decorators to be applied, stored outermost first (i.e. the first in the list will be applied last).
        # node.returns : the return annotation (Python 3 only).
        # node.type_comment : optional string containing the PEP 484 type comment of the function (added in Python 3.8)
        node = self.generic_visit(node)
        from icecream import ic

        assert_nodes = [
            self.make_assert_node(a.arg, a.annotation)
            for a in node.args.args
            if a.annotation
        ]

        new_node = ast.FunctionDef(
            node.name + "_typecheck_wrap",
            node.args,
            assert_nodes + node.body,
            node.decorator_list,
            node.returns,
            node.type_comment,
        )
        new_node = ast.copy_location(new_node, node)

        return new_node

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

        node_assert = self.make_assert_node(node.target.id, node.annotation)
        return [node, node_assert]


def typecheck(f, show_src=False, refers=()):
    """
    TODO: Sync this with README automatically.

    Decorator which turns annotated assignments of the form
      x : T = e
    into
      x : T = e
      assert isinstance(x, T), "x not of type T"

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

      def foo_typecheck_wrap(x: int, y: float):
        assert isinstance(x, int), 'x not of type int'
        assert isinstance(y, float), 'y not of type float'
        z: int = x * y
        assert isinstance(z, int), 'z not of type int'
        w: float = z * 3.2
        assert isinstance(w, float), 'w not of type float'
        return w

    If you want to see that transformed code, call with show_src=True

      @functools.partial(typecheck, show_src=True)
      def foo(x : int, y : float):
        z : int = x * y # Now it does
        w : float = z * 3.2
        return w

    """

    f_node, filename = get_ast_for_function(f)
    f_new_node = TypeCheckVisitor().visit(f_node)  # TODO unplumb empty list

    loc = lambda x: ast.copy_location(x, f_node)

    # Now add nodes for freevars
    #
    # Why ?  We tried just adding
    #     f_code = f_code.replace(co_freevars=f.__code__co_freevars)
    # and
    #     nonlocal_nodes = (
    #     [ast.copy_location(ast.Nonlocal(list(*self.freevars)), node)]
    #     if self.freevars
    #     else []
    # )
    # and other stuff
    # So...

    # Now add nodes declaring freevars and other needed names
    #  z_in_outer_scope = ...
    #  jnp = ...
    # These will be replaced by the freevars in f.__closure__, but
    # need to be in scope when compiling the AST
    # Function may refer to nonlocal names in its argument list,
    # which can't be declared 'nonlocal', as that declaration is
    # after the argument list, so we place fakes in scope

    f_freevars = f.__code__.co_freevars if f.__code__.co_freevars else ()
    f_closure = f.__closure__ if f.__closure__ else ()

    refers_names = tuple(r.__name__ for r in refers) + f_freevars
    # TODO: Do we need to worry about garbage collection of CellType?
    refers_closure = tuple(types.CellType(v) for v in refers) + f_closure

    refers_decls = [loc(ast.parse(f"{x} = ...").body[0]) for x in refers_names]
    new_node = ast.FunctionDef(
        name="_",
        args=[],
        body=refers_decls + f_new_node.body,
        decorator_list=[],
        posonlyargs=[],
        returns=None,
    )
    new_node = loc(new_node)
    ast.fix_missing_locations(new_node)
    new_node = ast.Module(body=[new_node], type_ignores=[])
    new_node = loc(new_node)

    if show_src or True:
        print("typecheck: Transformed source code")
        new_src = ast.unparse(new_node)
        print(new_src)

    # Compile new AST to get wrapped function
    try:
        if True:
            new_src = ast.unparse(new_node)
            new_code = compile(new_src, filename="TODO:linenos" + filename, mode="exec")
        else:
            new_code = compile(new_node, filename=filename, mode="exec")
    except Exception as e:
        # Most compile errors are pretty opaque (https://stackoverflow.com/a/25795966)
        # So call astpretty.  If it succeeds, it's helpful to debug, if it fails, its
        # error messages are much more helpful
        msg = astpretty.pformat(new_node)
        print(msg)
        raise ValueError("See AST printed above") from e

    konsts = new_code.co_consts[0].co_consts
    fns = tuple(k for k in konsts if isinstance(k, types.CodeType))
    assert len(fns) == 1, "TODO: Will need a better way to find the code"

    f_code = fns[0]

    f_name = f.__name__ + "_typecheck_wrap"
    f_checked = types.FunctionType(
        f_code,
        globals=f.__globals__,
        name=f_name,
        argdefs=f.__defaults__,
        closure=refers_closure,
    )
    f_checked.__wrapped__ = f
    return f_checked


def assert_is_instance(var, annot_type, varname, annot_str):
    if not isinstance(var, annot_type):
        raise TypeError(
            f"{varname} not of type {annot_str}, was {type(var)}, value {var}"
        )


# How much am I going to regret this?
# Trying to avoid having to import the module as well as just the
# typecheck function
typecheck.assert_is_instance = assert_is_instance
