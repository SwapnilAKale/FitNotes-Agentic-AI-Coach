"""
The chat's markdown rendering — frontend/index.html `md()`.

WHY THIS EXISTS. The analysis prompt has always instructed the model to use
"**bold** for key figures", "### " headers and "- " bullets to make an answer
scannable (analysis_agent.py, ANSWER FORMAT → READABILITY). Nothing in the
frontend ever rendered any of it: every agent reply went through esc() into a
`white-space: pre-wrap` bubble, so the user saw literal asterisks and hashes for
the entire life of the app. The model was formatting into a void.

TWO LAYERS OF TEST, deliberately:

  1. BEHAVIOURAL — the real thing. `md()` is extracted from index.html and run
     under node, so these assert what the browser will actually produce. Skipped
     with a clear reason when node is unavailable, rather than silently passing.
  2. STRUCTURAL — text guards that always run, pinning the properties a reader
     of the file must not undo (esc-before-decorate, no CDN, no single-* italics).

The safety argument is the ORDERING: esc() first, decorate the already-escaped
string second. Every tag is built from our own literals. test_escapes_* is what
holds that.
"""

import json
import shutil
import subprocess
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
INDEX = _ROOT / "frontend" / "index.html"

_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None,
                                 reason="node not installed — behavioural render tests skipped")


def _source() -> str:
    return INDEX.read_text(encoding="utf-8")


# Two spans, not one-per-helper. Picking helpers off individually broke twice:
# each new one landed between an anchor and its `\n}` terminator and silently
# truncated the capture, so md() ran with a missing dependency. `_ROW_RE`
# through the end of `md()` is one contiguous block in the file — take all of it.
_MD_PARTS = (
    r"function esc\(s\)[\s\S]*?\n\}",           # escaping
    r"const _ROW_RE[\s\S]*?\n  \}\n\}",         # every md helper + md itself
)


def _render(samples: list) -> list:
    """Run the REAL md() from index.html over each sample under node.

    Every helper md() depends on is pulled from the file too — extracting md()
    alone silently produced a ReferenceError once the table helpers were added.
    """
    src = _source()
    parts = []
    for pattern in _MD_PARTS:
        m = re.search(pattern, src)
        assert m, f"could not extract {pattern!r} from index.html"
        parts.append(m.group(0))
    script = ("\n".join(parts) + "\n"
              + "const input = JSON.parse(process.argv[1]);\n"
              + "process.stdout.write(JSON.stringify(input.map(md)));")
    out = subprocess.run([_NODE, "-e", script, json.dumps(samples)],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _one(sample: str) -> str:
    return _render([sample])[0]


# ── Behavioural: what the browser will actually show ─────────────────────────

@_needs_node
def test_bold_renders():
    assert _one("You did **82 sets** today.") == \
        "You did <strong>82 sets</strong> today."


@_needs_node
def test_headers_and_bullets_render():
    assert _one("### Summary") == '<span class="md-h">Summary</span>'
    assert _one("- 82 sets target") == '<span class="md-li">• 82 sets target</span>'
    assert _one("* 108 assisting") == '<span class="md-li">• 108 assisting</span>'


@_needs_node
def test_inline_code_renders():
    assert "<code" in _one("Use `straps` here.")


@_needs_node
@pytest.mark.parametrize("payload", [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<a href='javascript:alert(1)'>x</a>",
])
def test_escapes_html_before_decorating(payload):
    """The whole safety argument. No input may introduce markup."""
    out = _one(payload)
    assert "<script" not in out and "<img" not in out and "<a " not in out
    assert "&lt;" in out


@_needs_node
def test_escaping_survives_decoration():
    """Markup inside a bold span must still be inert."""
    out = _one("**<b>hi</b>**")
    assert out == "<strong>&lt;b&gt;hi&lt;/b&gt;</strong>"


_STRIP = re.compile(r"<[^>]+>")


@_needs_node
def test_display_set_content_survives_rendering():
    """Set lines are now DRAWN as a table (the user asked for it), so they are
    no longer byte-identical after rendering. What must survive is the CONTENT —
    every weight, rep count, label and comment.

    The verbatim contract itself is unaffected: enforce_display_fidelity compares
    the ANSWER TEXT, which this never touches. This is presentation only.
    """
    for line in ("Set 2 (Warmup): 60 kg × 8 reps",
                 "Set 1: 100 kg × 5 reps",
                 "Set 3: 5.0 km in 27:30"):
        plain = _STRIP.sub("", _one(line)).replace("&amp;", "&")
        for piece in re.findall(r"[\w.]+", line):
            assert piece in plain, f"{piece!r} lost from {line!r}"


@_needs_node
def test_stray_asterisks_and_underscores_are_left_alone():
    """THE NEGATIVE THAT MATTERS. Set comments are free text. A lone asterisk or
    underscore must never silently italicise half an answer — which is why
    single-* and _underscore_ italics are deliberately unsupported."""
    for text in ("Felt *heavy* today_ok on the last one",
                 "a * b * c",
                 "rest_pause set_2 done"):
        assert _one(text) == text


@_needs_node
def test_newlines_are_preserved_not_reflowed():
    out = _one("line one\nline two")
    assert out == "line one\nline two"


# ── Structural: properties a future edit must not undo ───────────────────────

def test_agent_replies_go_through_md_not_esc():
    src = _source()
    assert "addMsg('agent', esc(" not in src, \
        "an agent reply is still rendered esc()-only — its markdown will show as raw characters"
    assert "addMsg('agent', md(" in src


def _restore_history_src() -> str:
    m = re.search(r"function restoreHistory\(history\)[\s\S]*?\n\}", _source())
    assert m, "could not extract restoreHistory from index.html"
    return m.group(0)


def test_restored_agent_replies_go_through_md():
    """The RELOAD path. Live (2026-09-16) every answer showed raw ### and ** after
    a page reload: restoreHistory rendered every message with esc(), and the test
    above never saw it because it only looks for the literal `addMsg('agent', esc(`
    while restoreHistory writes `addMsg(role, ...)`. So look inside the function
    itself: esc() may only ever apply to the user's own text there."""
    code = [ln for ln in _restore_history_src().splitlines()
            if not ln.strip().startswith("//")]          # comments may say anything
    assert any("md(" in ln for ln in code), \
        "restoreHistory never renders markdown — reloads show raw ### and **"
    for ln in code:
        if "esc(" in ln:
            # Only the user's own text and a recorded ERROR are plain escaped
            # text; a coach answer is markdown. tests/test_reload_page.py checks
            # what each actually renders.
            assert "user" in ln or "error" in ln, \
                f"esc() on a coach-answer path in restoreHistory: {ln.strip()!r}"


@_needs_node
def test_reload_renders_agent_markdown_and_escapes_user_text():
    """Run the REAL restoreHistory, with addMsg stubbed to record what it is handed."""
    src = _source()
    parts = [re.search(p, src).group(0) for p in _MD_PARTS]
    script = ("\n".join(parts) + "\n"
              + "const calls = [];\n"
              + "function chatInner() { return {}; }\n"
              + "function addMsg(role, html, time) { calls.push([role, html]); }\n"
              + _restore_history_src() + "\n"
              + "restoreHistory(JSON.parse(process.argv[1]));\n"
              + "process.stdout.write(JSON.stringify(calls));")
    history = [{"role": "user", "text": "plan <b>my</b> week"},
               {"role": "assistant", "text": "### Your week\n* **Monday:** 4 sets"}]
    out = subprocess.run([_NODE, "-e", script, json.dumps(history)],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    (user_role, user_html), (agent_role, agent_html) = json.loads(out.stdout)
    assert (user_role, agent_role) == ("user", "agent")
    assert user_html == "plan &lt;b&gt;my&lt;/b&gt; week"
    assert '<span class="md-h">Your week</span>' in agent_html
    assert "<strong>Monday:</strong>" in agent_html
    assert "###" not in agent_html and "**" not in agent_html


def test_md_escapes_before_decorating():
    """Reversing these two steps would make every reply an injection point."""
    md_src = re.search(r"function md\(s\)[\s\S]*?\n  \}\n\}", _source()).group(0)
    assert "esc(" in md_src
    assert md_src.index("esc(") < md_src.index("<strong>")


def test_single_asterisk_italics_stay_unsupported():
    md_src = re.search(r"function md\(s\)[\s\S]*?\n  \}\n\}", _source()).group(0)
    assert "<em>" not in md_src and "<i>" not in md_src


def test_rendering_adds_no_network_dependency():
    """The graph viewer vendors three.js and is tested for reaching no network;
    the renderer must not undo that discipline by pulling in a markdown CDN."""
    src = _source()
    for bad in ("marked", "showdown", "markdown-it", "dompurify", "remarkable"):
        assert bad not in src.lower(), f"chat pulled in {bad}"


def test_bubble_keeps_pre_wrap():
    """md() decorates in place and relies on pre-wrap for layout; dropping it
    would collapse the whitespace of a verbatim display block."""
    assert re.search(r"\.bubble-agent\s*\{[^}]*white-space:\s*pre-wrap", _source())


# ── The prompt rules that make the rendered structure worth rendering ────────

def _prompt() -> str:
    """The COMPOSED analytical prompt — what the model is actually given.

    This used to read src/analysis_agent.py as raw source text. That was a proxy
    that stopped being true once the shared ADVICE STYLE / MEDICAL LINE blocks
    moved to src/prompt_blocks.py: the rules were still in the prompt, just no
    longer in that file. Worse than the one assertion that went red were the two
    that stayed green — `.split("ADVICE STYLE")` on text lacking the marker
    silently returns everything, so they began checking a wider region than they
    were written to check. Asserting on the composed value cannot drift that way.
    """
    from src.analysis_agent import _ANALYSIS_SYSTEM
    return _ANALYSIS_SYSTEM


def test_answer_format_leads_with_the_users_problem():
    """Live, "my grip gave out on shrugs" got three sentences about the grip not
    being trained, and the actual advice — straps, reorder the session — in the
    LAST paragraph. The muscle-role fact is the reason, not the answer."""
    p = _prompt()
    assert "LEAD WITH THEIR PROBLEM" in p
    assert "FIRST paragraph answers THEIR" in p
    assert "most important finding FOR THE QUESTION ASKED" in p


def test_a_performance_limit_is_not_a_medical_symptom():
    """The same answer ended with "consider if a professional assessment is
    needed" — off a set count. The medical redirect is earned by pain or injury,
    not by a muscle fatiguing, which is what muscles do."""
    p = _prompt()
    assert "A PERFORMANCE LIMIT IS NOT A SYMPTOM" in p
    assert "grip gave out" in p


def test_a_report_of_how_a_lift_went_routes_analytical():
    """Live, "my grip gave out on shrugs" was classified OPERATIONAL and the
    agent offered to save an exercise-quirk note instead of answering. The
    analytical list already names "failed attempts"; the prompt just never said
    that a report of how a set went is not a write request."""
    src = (_ROOT / "src" / "coordinator.py").read_text(encoding="utf-8")
    assert "A REPORT OF HOW A LIFT WENT IS ANALYTICAL, NOT A WRITE" in src
    assert "my grip gave out on shrugs" in src


def test_html_is_served_no_cache():
    """A cached index.html is indistinguishable from code that did not change —
    it made the markdown renderer look broken while the server was serving it."""
    src = (_ROOT / "frontend" / "server.py").read_text(encoding="utf-8")
    assert "no-cache" in src
    assert src.count("headers=_NO_CACHE") >= 2, "both / and /graph must revalidate"


# ── Tables ───────────────────────────────────────────────────────────────────
#
# The one construct that cannot be decorated in place: a real <table> cannot be
# built line-by-line inside `white-space: pre-wrap`. So a table is detected as a
# BLOCK and replaced wholesale, and everything else keeps the old path. A week
# plan used to reach the user as raw pipes and dashes.

_TABLE = ("| Day | Focus | Exercises |\n"
          "|:--- |:---:| ---:|\n"
          "| 1 | Chest | **Flat DB Bench** (3 sets) |\n"
          "| 2 | Back | Lat Pulldown |")


@_needs_node
def test_markdown_table_renders_as_a_table():
    out = _one(_TABLE)
    assert "<table" in out and "<thead>" in out
    assert out.count("<tr>") == 3          # header + 2 body rows
    assert "|" not in out                  # no raw pipes survive


@_needs_node
def test_table_honours_column_alignment():
    out = _one(_TABLE)
    assert 'text-align:center' in out      # :---:
    assert 'text-align:right' in out       # ---:


@_needs_node
def test_table_cells_keep_inline_decoration():
    assert "<strong>Flat DB Bench</strong>" in _one(_TABLE)


_BLOCK = ("2026-06-24 — Smith Machine Shrugs (4 sets):\n"
          "Set 1: 154.1 lbs × 12 reps\n"
          "Set 2: 164.1 lbs × 11 reps (Last one not fully squeezed at the top)\n"
          "Set 3: 174.1 lbs × 4 reps (Last one not fully squeezed at the top)\n"
          "       134.1 lbs × 8 reps")


@_needs_node
def test_display_block_renders_as_a_table():
    out = _one(_BLOCK)
    assert "<table" in out and "md-sets" in out
    assert out.count("<tr>") == 3            # three sets, not four
    assert "Smith Machine Shrugs" in out     # header kept


@_needs_node
def test_a_drop_set_stays_inside_its_own_row():
    """The visible bug: the drop-set continuation arrives inside Set 3's string
    as an indented second line, and in a pre-wrap bubble it read as a broken
    fourth set. It belongs in Set 3's cell."""
    out = _one(_BLOCK)
    third = out.split("<tr>")[-1]
    assert "174.1" in third and "134.1" in third


@_needs_node
def test_set_comments_move_to_their_own_column():
    out = _one(_BLOCK)
    assert 'md-set-note">Last one not fully squeezed at the top' in out


@_needs_node
def test_display_block_content_round_trips():
    """Nothing may be LOST in the table — every number and word still present."""
    plain = _STRIP.sub("", _one(_BLOCK))
    for piece in ("154.1", "12", "164.1", "11", "174.1", "4", "134.1", "8",
                  "Smith Machine Shrugs", "Last one not fully squeezed"):
        assert piece in plain, f"{piece!r} lost"


@_needs_node
def test_prose_mentioning_sets_is_not_a_table():
    for text in ("You did 3 sets today.", "Set a goal for 150 lbs."):
        assert "<table" not in _one(text)


@_needs_node
def test_a_separator_row_is_required():
    """A sentence that merely contains pipes is not a table."""
    for text in ("Use straps | or not, your call.",
                 "| a | b |\n| c | d |"):
        assert "<table" not in _one(text)


@_needs_node
def test_text_around_a_table_is_untouched():
    out = _one("Here is the week:\n" + _TABLE + "\nMonitor your recovery.")
    assert out.startswith("Here is the week:")
    assert out.rstrip().endswith("Monitor your recovery.")
    assert "<table" in out


# ── Plan rules in the prompt ─────────────────────────────────────────────────

def test_plan_rules_cover_scope_frequency_and_completeness():
    """The prompt had extensive rules on not lying about numbers and NOTHING on
    what a training plan must contain — which is why a plan dropped every other
    muscle, scheduled biceps on back-to-back days, and listed two exercises."""
    p = _prompt()
    assert "TRAINING PLAN RULES" in p
    assert "SCOPE FORK" in p
    assert "CONSECUTIVE DAYS" in p
    assert "A SESSION IS A SESSION" in p


def test_the_consecutive_days_rule_gives_way_to_an_explicit_request():
    """A user may WANT the same muscle every day. The rule used to say NEVER with
    no exception, so the coach (and the plan guard) overrode a split the user
    asked for on purpose (live re-check, 2026-09-16)."""
    rule = _prompt().split("CONSECUTIVE DAYS")[1].split("VOLUME IN SETS")[0]
    assert "explicitly asks" in rule


def test_plan_rule_examples_avoid_the_live_probes():
    """Discipline from PROMPT-EXAMPLE-AUDIT.md: teach on one instance, test on
    another. The plan section must not name the muscles used to verify it, or
    the live check proves only that the model can copy."""
    section = _prompt().split("TRAINING PLAN RULES")[1].split("ADVICE STYLE")[0]
    for probe_word in ("calve", "rear delt", "chest", "hamstring", "quad"):
        assert probe_word not in section.lower(), \
            f"plan examples mention {probe_word!r}, which a live probe also uses"


def test_prompt_prefers_sets_per_week_over_volume():
    """Volume multiplies load by reps, so the same dose reads differently at 5
    reps than 15 and compares to nothing. Sets per week is the dose measure —
    and until now it was the only view the package did NOT have."""
    p = _prompt()
    assert "SETS PER WEEK IS THE DEFAULT MEASURE" in p
    assert "primary_sets_per_week" in p
    assert "VOLUME RULES" in p, "volume guidance must survive for volume questions"


def test_prompt_binds_exercise_names_to_the_store():
    """It invented "Face Pulls" for an exercise the store holds as Cable Face
    Pull. Two lists, nothing outside them."""
    p = _prompt()
    assert "EXERCISE NAMES COME FROM THE STORE" in p
    assert "suggestable_exercises" in p


def test_plan_rules_size_volume_in_sets_not_pounds():
    p = _prompt()
    section = p.split("TRAINING PLAN RULES")[1].split("ADVICE STYLE")[0]
    assert "primary_sets_per_week" in section
