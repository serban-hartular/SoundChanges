import re
from typing import Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


DOOM3_SEARCH_URL = "https://doom.lingv.ro/cautare/q/{}"
NOUN_MARKER = re.compile(r"\bs\.\s*[mfn]\.")
PLURAL_MARKER = re.compile(r"\bpl\.")
UNDERLINED = re.compile(r"text-decoration\s*:\s*underline", re.IGNORECASE)


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

    for node in lemma_element.next_siblings:
        if isinstance(node, NavigableString):
            text = str(node)
            if not in_plural:
                if PLURAL_MARKER.search(text):
                    in_plural = True
            elif re.search(r"[,;]", text):
                # A slash separates alternatives; punctuation starts the next
                # paradigm field (usually an articulated form).
                break
        elif isinstance(node, Tag) and in_plural:
            if "font-style: italic" in node.get("style", ""):
                plurals.append(node)

    return plurals


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

        plurals = _plural_elements(entry, lemma_element)
        if not plurals:
            results.append(
                {
                    "form": form,
                    "stress": stress,
                    "plural": None,
                    "plural_stress": None,
                }
            )
            continue

        for plural_element in plurals:
            plural, plural_stress = _form_and_stress(plural_element)
            results.append(
                {
                    "form": form,
                    "stress": stress,
                    "plural": plural,
                    "plural_stress": plural_stress,
                }
            )

    return results


if __name__ == "__main__":
    for example in ["praștie", "plasă", "nivel", "albastru"]:
        print(example, extract_noun(example))
