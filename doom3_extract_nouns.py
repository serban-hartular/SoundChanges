import re
from typing import Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


DOOM3_SEARCH_URL = "https://doom.lingv.ro/cautare/q/{}"
ROMANIAN_LOWERCASE_LETTERS = frozenset("abcdefghijlmnoprstuvxzăîâșț")
NOUN_MARKER = re.compile(r"\bs\.\s*[mfn]\.")
PLURAL_MARKER = re.compile(r"\bpl\.")
UNDERLINED = re.compile(r"text-decoration\s*:\s*underline", re.IGNORECASE)
LANGUAGE_INDICATIONS = (
    "engl. / fr.",
    "port., sp.",
    "cuv. amerind.",
    "cuv. polinez.",
    "cuv. chin.",
    "cuv. jap.",
    "lat. șt.",
    "cuv. ar.",
    "cuv. ebr.",
    "cuv. cor.",
    "engl.",
    "fr.",
    "lat.",
    "germ.",
    "it.",
    "sp.",
    "gr.",
    "port.",
    "rus.",
    "sanscr.",
    "chin.",
    "afr.",
    "neerl.",
)


def filter_nouns(strings: list[str]) -> list[str]:
    """Normalize and filter noun candidates, preserving their input order.

    Trailing ASCII digits and surrounding whitespace are removed first. Valid
    results contain at least two lowercase Romanian-alphabet letters and no
    whitespace, dashes, uppercase letters, or q/w/y/k.
    """
    results: list[str] = []
    seen: set[str] = set()

    for item in strings:
        noun = re.sub(r"[0-9]+$", "", item.strip())
        if (
            len(noun) < 2
            or any(letter not in ROMANIAN_LOWERCASE_LETTERS for letter in noun)
            or noun in seen
        ):
            continue
        seen.add(noun)
        results.append(noun)

    return results


def _form_and_stress(element: Tag) -> tuple[str, Optional[int]]:
    """Return an element's text and the index of its underlined character."""
    form = element.get_text("", strip=True)
    underlined = element.find(
        lambda tag: isinstance(tag, Tag)
        and UNDERLINED.search(tag.get("style", "")) is not None
    )
    if underlined is None:
        return form, None

    stressed_text = underlined.get_text("", strip=True)
    prefix = ""
    for descendant in element.descendants:
        if descendant is underlined:
            break
        if isinstance(descendant, NavigableString):
            prefix += str(descendant)

    return form, len(prefix) if stressed_text else None


def _plural_elements(entry: Tag, lemma_element: Tag) -> list[Tag]:
    """Find the indefinite plural form elements following the `pl.` marker."""
    plurals: list[Tag] = []
    in_plural = False
    text_after_form = ""

    for node in lemma_element.next_siblings:
        if isinstance(node, NavigableString):
            text = str(node)
            if not in_plural:
                marker = PLURAL_MARKER.search(text)
                if marker:
                    in_plural = True
                    text_after_form = text[marker.end() :]
            else:
                text_after_form += text
        elif isinstance(node, Tag) and in_plural:
            if "font-style: italic" in node.get("style", ""):
                # The first italic element after `pl.` is the plural. Further
                # italic elements are alternatives only when separated from
                # the previous form by a slash. This excludes pronunciation
                # and syllabification annotations, which are also italicized.
                if plurals and re.fullmatch(r"\s*/\s*", text_after_form) is None:
                    break
                plurals.append(node)
                text_after_form = ""
            elif plurals:
                break

    return plurals


def _language_indication(entry: Tag) -> Optional[str]:
    """Return a recognized parenthesized language indication, without parens."""
    # Whitespace is presentation-only on DOOM pages and may include NBSPs and
    # line breaks around the semantic <lang> element.
    compact_text = re.sub(r"\s+", "", entry.get_text("", strip=True)).casefold()
    for indication in LANGUAGE_INDICATIONS:
        compact_indication = re.sub(r"\s+", "", indication).casefold()
        if f"({compact_indication})" in compact_text:
            return indication
    return None


def _singular_syllabification(lemma_element: Tag) -> Optional[str]:
    """Return DOOM's singular syllabification annotation, when present."""
    syllabification = None
    for node in lemma_element.next_siblings:
        if isinstance(node, NavigableString):
            if NOUN_MARKER.search(str(node)):
                return syllabification
        elif isinstance(node, Tag):
            if node.get("title") == "despărțirea in silabe":
                syllabification = node.get_text("", strip=True)

    return None


def _plural_syllabification(plural_element: Tag) -> Optional[str]:
    """Return the syllabification annotation immediately after a plural."""
    for node in plural_element.next_siblings:
        if isinstance(node, NavigableString):
            if re.search(r"[/,;]", str(node)):
                return None
        elif isinstance(node, Tag):
            if node.get("title") == "despărțirea in silabe":
                return node.get_text("", strip=True)
            return None

    return None


def extract_noun(noun: str, *, timeout: float = 30) -> list[dict]:
    """Download a DOOM 3 page and extract exact-match noun paradigms.

    One dictionary is returned for every plural alternative. A noun without a
    plural produces one dictionary whose ``plural`` and ``plural_stress`` are
    both ``None``. Stress indices are zero-based character indices; if DOOM
    does not underline a vowel, the corresponding stress value is ``None``.
    """
    noun = noun.strip()
    url = DOOM3_SEARCH_URL.format(quote(noun, safe=""))
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "SoundChanges/1.0 (DOOM3 noun extractor)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict] = []

    for entry in soup.find_all("entry"):
        lemma_element = entry.find(
            "span", style=lambda value: value and "font-weight: bold" in value
        )
        if lemma_element is None:
            continue

        form, stress = _form_and_stress(lemma_element)
        # Homonym numbers are included in the lemma span as superscripts.
        superscript = lemma_element.find("sup")
        if superscript is not None:
            form = form.removesuffix(superscript.get_text("", strip=True))

        if form.casefold() != noun.casefold():
            continue

        entry_text = entry.get_text(" ", strip=True)
        lemma_text = lemma_element.get_text(" ", strip=True)
        text_after_lemma = entry_text.removeprefix(lemma_text)
        if NOUN_MARKER.search(text_after_lemma) is None:
            continue

        lang = _language_indication(entry)
        syll_sg = _singular_syllabification(lemma_element)
        plurals = _plural_elements(entry, lemma_element)
        if not plurals:
            results.append(
                {
                    "form": form,
                    "stress": stress,
                    "plural": None,
                    "plural_stress": None,
                    "lang": lang,
                    "syll_sg": syll_sg,
                    "syll_pl": None,
                }
            )
            continue

        for plural_element in plurals:
            plural, plural_stress = _form_and_stress(plural_element)
            syll_pl = _plural_syllabification(plural_element)
            results.append(
                {
                    "form": form,
                    "stress": stress,
                    "plural": plural,
                    "plural_stress": plural_stress,
                    "lang": lang,
                    "syll_sg": syll_sg,
                    "syll_pl": syll_pl,
                }
            )

    return results


if __name__ == "__main__":
    from pathlib import Path
    import random

    nouns = Path("lang_data/nouns.doom.lingv.ro.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    filtered_nouns = filter_nouns(nouns)
    noun_entries = []
    errors = []
    for i, noun in enumerate(filtered_nouns):
        print(i+1, '/', len(filtered_nouns), '\t', noun, end='\t')
        try:
            d_list = extract_noun(noun)
            noun_entries.extend(d_list)
            print('ok')
        except Exception as e:
            errors.append((noun, e))
            print('err')

    import pickle
    with open('./lang_data/doom3.nouns.v0.p', 'wb') as handle:
        pickle.dump(noun_entries, handle)

    # for example in ["praștie", "plasă", "nivel", "albastru"]:
    #     print(example, extract_noun(example))
