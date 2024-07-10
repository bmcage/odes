from scikits_odes_sundials import _get_num_args


class TestGetNumArgs:
    """
    Check the correct number for `_get_num_args`
    """
    def test_functions(self):
        def _func(a, b, c):
            pass
        assert _get_num_args(_func) == 3

        def _func(a, b, c=None):
            pass
        # kwd args should not make a difference
        assert _get_num_args(_func) == 3

    def test_methods(self):
        class C:
            @ classmethod
            def class_method(cls, a, b):
                pass

            @ staticmethod
            def static_method(a, b):
                pass

            def method(self, a, b):
                pass

        # self should not be counted
        assert _get_num_args(C.method) == 2
        # static methods should also work
        assert _get_num_args(C.static_method) == 2
        # class methods should also work
        assert _get_num_args(C.class_method) == 2
