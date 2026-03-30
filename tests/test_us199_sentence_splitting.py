"""Tests for US-199: abbreviation-aware sentence splitting in the TTS pipeline."""

from __future__ import annotations


class TestSplitIntoSentencesAbbreviations:
    """_split_into_sentences must not break on common abbreviations."""

    def _split(self, text: str) -> list[str]:
        # Import here so pytest collection does not trigger heavy imports.
        from rex.voice_loop import _split_into_sentences

        return _split_into_sentences(text)

    def test_dr_title_is_single_sentence(self):
        """'Dr. Smith said the treatment works.' must be one sentence."""
        result = self._split("Dr. Smith said the treatment works.")
        assert result == ["Dr. Smith said the treatment works."]

    def test_mr_title_is_single_sentence(self):
        result = self._split("Mr. Jones agreed with the plan.")
        assert result == ["Mr. Jones agreed with the plan."]

    def test_eg_abbreviation_is_single_sentence(self):
        """'e.g. this example.' must be treated as one sentence."""
        result = self._split("e.g. this example.")
        assert result == ["e.g. this example."]

    def test_etc_abbreviation_stays_in_sentence(self):
        result = self._split("Bring items etc. and meet us there.")
        assert result == ["Bring items etc. and meet us there."]

    def test_two_real_sentences_split_correctly(self):
        """'She said it was great. He agreed.' must split into two sentences."""
        result = self._split("She said it was great. He agreed.")
        assert len(result) == 2
        assert result[0] == "She said it was great."
        assert result[1] == "He agreed."

    def test_question_and_exclamation_split(self):
        result = self._split("Is this right? Yes it is!")
        assert len(result) == 2

    def test_mixed_abbreviation_and_real_split(self):
        """Dr. prefix should not split; period after full sentence should split."""
        result = self._split("Dr. Smith arrived. He was late.")
        assert len(result) == 2
        assert result[0] == "Dr. Smith arrived."
        assert result[1] == "He was late."

    def test_empty_string(self):
        assert self._split("") == []

    def test_single_word(self):
        result = self._split("Hello.")
        assert result == ["Hello."]

    def test_whitespace_stripped(self):
        result = self._split("  Hello.   World.  ")
        assert all(s == s.strip() for s in result)
