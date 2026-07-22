"""Tests for the calculator tool, especially that it rejects anything beyond
plain arithmetic -- it parses expressions with ast, not eval(), specifically
so model-supplied input can never reach arbitrary code execution."""
import pytest

from tools import calculator


class TestCalculatorArithmetic:
    def test_addition(self):
        assert calculator("2 + 3") == "5"

    def test_operator_precedence(self):
        assert calculator("2 + 3 * 4") == "14"

    def test_parentheses(self):
        assert calculator("(2 + 3) * 4") == "20"

    def test_division(self):
        assert calculator("7 / 2") == "3.5"

    def test_power(self):
        assert calculator("2 ** 10") == "1024"

    def test_modulo(self):
        assert calculator("10 % 3") == "1"

    def test_negative_numbers(self):
        assert calculator("-5 + 3") == "-2"

    def test_decimal_numbers(self):
        assert calculator("(49.99 * 2) - 15") == "84.98"


class TestCalculatorRejectsUnsafeInput:
    def test_rejects_name_lookup(self):
        assert calculator("x + 1").startswith("Error:")

    def test_rejects_function_call(self):
        assert calculator("__import__('os').system('echo hi')").startswith("Error:")

    def test_rejects_attribute_access(self):
        assert calculator("().__class__").startswith("Error:")

    def test_rejects_string_literal(self):
        assert calculator("'a' + 'b'").startswith("Error:")

    def test_rejects_garbage_input(self):
        assert calculator("not even math").startswith("Error:")
