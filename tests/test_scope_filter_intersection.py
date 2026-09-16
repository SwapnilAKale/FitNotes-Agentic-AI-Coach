"""
exercise_names and muscle_groups must NOT narrow each other.

THE DEFECT. Both were applied as row filters, ANDed:

    if muscle_groups:   rows = [r for r in rows if r["category_id"] in allowed]
    if exercise_names:  rows = [r for r in rows if r["exercise_name"] in names]

"Do my lat pulldowns train my biceps?" yields exercise_names=['Lat Pulldown']
and muscle_groups=['Biceps']. FitNotes files Lat Pulldown under BACK, so the
conjunction was EMPTY BY CONSTRUCTION — for every window, every user, forever.
The package fell back to broad scope and the agent invented an answer about an
exercise with 366 logged sets, reporting it as absent from the history.

The sharp edge: one-category-per-exercise is the exact FitNotes limitation the
muscle ontology was built to work around, and the cross-category question was
being killed BY that category before the ontology was ever consulted.

THE RULE. When exercise_names is present it IS the scope; a muscle group named
in the same sentence is the SUBJECT of the question, not a second narrowing.

Pinned as an INVARIANT over many groups rather than as the one instance that
was reported — an example is one sample of a class.
"""

import pytest

from src.data_agent import collect

_GROUPS = ["Back", "Chest", "Shoulders", "Biceps", "Triceps", "Legs",
           "Forearms", "Abs"]


def _names(**kw):
    return sorted(e.get("name") for e in
                  collect(query_period_days=365, **kw).get("exercises", []))


@pytest.fixture(scope="module")
def subject():
    """A real logged exercise to filter on — prefers the one from the live
    report so the original defect stays literally pinned, but falls back to
    whatever is logged so the test is not hostage to one person's data."""
    logged = _names()
    if not logged:
        pytest.skip("no logged exercises in the database")
    return "Lat Pulldown" if "Lat Pulldown" in logged else logged[0]


def test_a_muscle_group_never_narrows_an_exercise_filter(subject):
    """The invariant. Adding ANY muscle group to an exercise filter must not
    change what resolves — including the group the exercise is not filed under,
    which is the case that produced the empty set."""
    alone = _names(exercise_names=[subject])
    assert alone, f"{subject!r} should resolve on its own"

    for group in _GROUPS:
        both = _names(exercise_names=[subject], muscle_groups=[group])
        assert both == alone, (
            f"muscle_groups={group!r} changed an exercise-scoped query: "
            f"{alone} -> {both}")


def test_the_reported_case_resolves(subject):
    """The literal live failure, when the data supports it."""
    if subject != "Lat Pulldown":
        pytest.skip("Lat Pulldown not logged in this database")
    got = _names(exercise_names=["Lat Pulldown"], muscle_groups=["Biceps"])
    assert got == ["Lat Pulldown"]


def test_muscle_group_alone_still_filters():
    """The negative: removing the conjunction must not disable group filtering.
    A group query has to stay narrower than no filter at all."""
    broad = _names()
    grouped = _names(muscle_groups=["Biceps"])
    assert grouped, "a group filter should still resolve something"
    assert set(grouped) < set(broad), "group filter no longer narrows anything"


def test_no_filter_is_unchanged():
    assert len(_names()) > 1
