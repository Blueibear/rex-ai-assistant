"""Unit tests for rex.model_router — intent classification heuristics."""

import pytest

from rex.model_router import ModelRouter, TaskCategory


@pytest.fixture
def router() -> ModelRouter:
    return ModelRouter()


class TestCodingClassification:
    def test_write_function(self, router):
        assert router.classify("Write a function to sort a list") == TaskCategory.coding

    def test_debug(self, router):
        assert router.classify("Help me debug this Python script") == TaskCategory.coding

    def test_refactor(self, router):
        assert router.classify("Can you refactor this code?") == TaskCategory.coding

    def test_python(self, router):
        assert router.classify("How do I read a file in Python?") == TaskCategory.coding

    def test_sql(self, router):
        assert router.classify("Write a SQL query to join two tables") == TaskCategory.coding

    def test_unit_test(self, router):
        assert router.classify("Write a unit test for this class") == TaskCategory.coding


class TestReasoningClassification:
    def test_analyze(self, router):
        assert (
            router.classify("Analyze the pros and cons of this approach") == TaskCategory.reasoning
        )

    def test_step_by_step(self, router):
        assert router.classify("Walk me through this step by step") == TaskCategory.reasoning

    def test_complex_plan(self, router):
        assert (
            router.classify("Help me plan a complex migration strategy") == TaskCategory.reasoning
        )

    def test_compare(self, router):
        assert (
            router.classify("Compare these two frameworks and explain the tradeoffs")
            == TaskCategory.reasoning
        )

    def test_evaluate(self, router):
        assert (
            router.classify("Evaluate the best approach to scaling this service")
            == TaskCategory.reasoning
        )


class TestSearchClassification:
    def test_search_for(self, router):
        assert router.classify("Search for the latest news on AI") == TaskCategory.search

    def test_look_up(self, router):
        assert router.classify("Look up the current price of gold") == TaskCategory.search

    def test_find_information(self, router):
        assert router.classify("Find me information about Rex AI") == TaskCategory.search

    def test_web_search(self, router):
        assert (
            router.classify("Do a web search for the best coffee shops near me")
            == TaskCategory.search
        )


class TestVisionClassification:
    def test_describe_image(self, router):
        assert router.classify("Describe this image please") == TaskCategory.vision

    def test_what_is_in_photo(self, router):
        assert router.classify("What is in the photo?") == TaskCategory.vision

    def test_screenshot(self, router):
        assert (
            router.classify("Look at this screenshot and tell me what you see")
            == TaskCategory.vision
        )

    def test_ocr(self, router):
        assert router.classify("Use OCR to read the text in this image") == TaskCategory.vision


class TestFastClassification:
    def test_hello(self, router):
        assert router.classify("Hello") == TaskCategory.fast

    def test_what_time(self, router):
        assert router.classify("What time is it?") == TaskCategory.fast

    def test_what_date(self, router):
        assert router.classify("What is today's date?") == TaskCategory.fast

    def test_thanks(self, router):
        assert router.classify("Thanks!") == TaskCategory.fast

    def test_ok(self, router):
        assert router.classify("Ok got it") == TaskCategory.fast


class TestDefaultClassification:
    def test_unrecognized(self, router):
        assert router.classify("Tell me a story about a dragon") == TaskCategory.default

    def test_empty(self, router):
        assert router.classify("") == TaskCategory.default


class TestTaskCategoryIsStrEnum:
    def test_str_values(self):
        assert str(TaskCategory.coding) == "coding"
        assert str(TaskCategory.reasoning) == "reasoning"
        assert str(TaskCategory.search) == "search"
        assert str(TaskCategory.vision) == "vision"
        assert str(TaskCategory.fast) == "fast"
        assert str(TaskCategory.default) == "default"

    def test_equality_with_string(self):
        assert TaskCategory.coding == "coding"
