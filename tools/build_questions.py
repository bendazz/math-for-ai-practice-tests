"""
build_questions.py — the ONE source of truth for all 50 practice questions.

Run it after any edit:

    python3 tools/build_questions.py

It checks every answer, then writes ../questions.js (the file the site loads).
Never edit questions.js by hand; it gets overwritten.

What gets checked:
  * vector / dot / cosine / precision-recall answers are recomputed here in
    Python and asserted against the answer key;
  * every chunking question is re-run through the splitting algorithm, and —
    if you run it with the Langflow Desktop Python, which has langchain
    installed — through the REAL CharacterTextSplitter as well:

        ~/.langflow/.langflow-venv/bin/python tools/build_questions.py

  * each question has exactly 4 choices, one correct, no duplicates;
  * answer letters are spread evenly across A–D.

Question format
  choices: list of (text, why). The FIRST entry is the correct answer and its
           `why` is None. Every other entry's `why` explains the specific
           mistake that produces it (shown on the solutions page only).
  pos:     the letter the correct answer should land on. The build moves the
           correct choice there and keeps the distractors in listed order.
  doc:     (chunking only) (text, separator, chunk_size, chunk_overlap,
           expected_chunk_lengths). The page shows the document with a
           character count beside every line, like the textbook does.

All numbers are chosen to be doable with pencil and paper — no calculator.
"""

import json, math, os, sys
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "questions.js")

N, NN = "\n", "\n\n"


# ------------------------------------------------------------------ helpers
def fr(a, b):
    """Stacked fraction for display."""
    return f'<span class="fr"><span>{a}</span><span>{b}</span></span>'


def v(*xs):
    """Vector for display, with a true minus sign."""
    return "(" + ", ".join(str(x).replace("-", "−") for x in xs) + ")"


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm2(a):
    return dot(a, a)


def pct(num, den):
    return round(100 * num / den)


def merge(text, sep, size, overlap):
    """langchain CharacterTextSplitter._merge_splits, keep_separator=False."""
    atoms = [a for a in text.split(sep) if a != ""]
    sl, cur, total, chunks = len(sep), [], 0, []
    for a in atoms:
        n = len(a)
        if total + n + (sl if cur else 0) > size and cur:
            chunks.append(sep.join(cur).strip())
            while total > overlap or (total + n + (sl if cur else 0) > size and total > 0):
                total -= len(cur[0]) + (sl if len(cur) > 1 else 0)
                cur = cur[1:]
        cur.append(a)
        total += n + (sl if len(cur) > 1 else 0)
    chunks.append(sep.join(cur).strip())
    return [c for c in chunks if c]


try:
    import logging
    logging.disable(logging.CRITICAL)
    from langchain_text_splitters import CharacterTextSplitter
    HAVE_LANGCHAIN = True
except Exception:
    HAVE_LANGCHAIN = False


def check_chunks(text, sep, size, overlap, expected):
    got = [len(c) for c in merge(text, sep, size, overlap)]
    assert got == expected, f"chunk lengths {got} != expected {expected}"
    if HAVE_LANGCHAIN:
        real = CharacterTextSplitter(separator=sep, chunk_size=size, chunk_overlap=overlap,
                                     keep_separator=False).split_text(text)
        assert [len(c) for c in real] == expected, f"REAL splitter gave {[len(c) for c in real]}"


def shared_lines(text, sep, size, overlap):
    ch = merge(text, sep, size, overlap)
    return [len(set(ch[i].split(sep)) & set(ch[i - 1].split(sep))) for i in range(1, len(ch))]


# ------------------------------------------------------------------ topics
VEC = "Vector arithmetic & length"
DOT = "Dot product & cosine similarity"
CHK = "Chunking by hand"
EDGE = "Chunking edge cases"
PR = "Precision & recall in search"

Q = []  # every question, in test order


def q(test, topic, pos, prompt, choices, solution, doc=None, settings=None):
    Q.append(dict(test=test, topic=topic, pos=pos, prompt=prompt, choices=choices,
                  solution=solution, doc=doc, settings=settings))


# ================================================================== TEST 1
assert tuple(3 * x + y for x, y in zip((2, -1), (-4, 5))) == (2, 2)
q(1, VEC, "B", "<p>Compute 3(2, −1) + (−4, 5).</p>",
  [(v(2, 2), None),
   (v(-2, 4), "Forgot to scale. (2, −1) + (−4, 5) = (−2, 4) — but the 3 multiplies every component of (2, −1) first."),
   (v(2, 8), "A sign slip in the second component: 3 × (−1) = −3, and −3 + 5 = 2, not 8."),
   (v(10, -8), "Subtracted instead of added: (6, −3) − (−4, 5) = (10, −8).")],
  "<p>Scale first: 3(2, −1) = (6, −3).</p><p>Then add matching components: (6 + (−4), −3 + 5) = <strong>(2, 2)</strong>.</p>")

assert math.isqrt(norm2((6, -8))) == 10
q(1, VEC, "A", "<p>Find the length ‖(6, −8)‖.</p>",
  [("10", None),
   ("14", "That is 6 + 8 — adding the sizes of the components. Length squares them first, like Pythagoras."),
   ("100", "That is 6² + (−8)² — the right sum, but the square root step is missing."),
   ("2", "That is |6 + (−8)| — adding the components with their signs.")],
  "<p>Square, add, square-root: √(6² + (−8)²) = √(36 + 64) = √100 = <strong>10</strong>.</p>"
  "<p>Squaring makes the minus sign disappear: (−8)² = 64.</p>")

assert dot((4, -1), (2, 3)) == 5
q(1, DOT, "C", "<p>Compute the dot product (4, −1) · (2, 3).</p>",
  [("5", None),
   ("11", "A sign slip: (−1)(3) = −3, so it is 8 − 3, not 8 + 3."),
   ("(8, −3)", "Multiplying matching components is only half the job — then add them up. A dot product is a single number, not a vector."),
   ("15", "That is (4 + (−1)) × (2 + 3) — adding inside each vector first. Multiply matching components, then add.")],
  "<p>Multiply matching components and add: (4)(2) + (−1)(3) = 8 − 3 = <strong>5</strong>.</p>")

assert F(dot((3, 4), (4, 3)), 25) == F(24, 25) == F(96, 100)
q(1, DOT, "D", "<p>Find the cosine similarity of (3, 4) and (4, 3).</p>",
  [("0.96", None),
   ("24", "That is the dot product. Cosine similarity then divides by both lengths."),
   ("2.4", "That is 24 ÷ (5 + 5). The two lengths get <em>multiplied</em>, not added."),
   ("1", "Same numbers, different order — but they point in different directions, so the score is a little below 1.")],
  f"<p>Dot product: (3)(4) + (4)(3) = 24.</p>"
  f"<p>Lengths: ‖(3, 4)‖ = √25 = 5 and ‖(4, 3)‖ = 5.</p>"
  f"<p>Cosine similarity = {fr('24', '5 × 5')} = {fr(24, 25)} = <strong>0.96</strong>. "
  f"(To get the decimal by hand: {fr(24, 25)} = {fr(96, 100)}.) Very similar directions, but not identical.</p>")

T = "Doors open at noon.\nSign the guest book.\nFind a seat up front.\nPhones on silent."
q(1, CHK, "A", "<p>Give the chunks and their lengths.</p>",
  [("Two chunks: 40 and 39", None),
   ("Two chunks: 39 and 38", "Forgot the separator. Gluing two lines together costs one newline: 19 + 20 + 1 = 40."),
   ("Two chunks: 62 and 17", "Chunk 1 went past the Chunk Size. 40 + 21 + 1 = 62 is over 50, so line 3 has to start a new chunk."),
   ("Four chunks: 19, 20, 21, 17", "The separator decides the atoms, not the chunks. Atoms get glued together until the next one won't fit.")],
  "<p>Atoms: 19, 20, 21, 17.</p>"
  "<p>Buffer 19. Line 2: 19 + 20 + 1 = 40, not over 50 — it fits.</p>"
  "<p>Line 3: 40 + 21 + 1 = 62, over 50. <strong>Emit chunk 1 = 40.</strong> Overlap is 0, so pop everything. Buffer takes line 3: 21.</p>"
  "<p>Line 4: 21 + 17 + 1 = 39, fits. End of text: <strong>emit chunk 2 = 39.</strong></p>",
  doc=(T, N, 50, 0, [40, 39]))

T = ("The library opens at eight.\nStudy rooms must be booked.\nPrinting costs ten cents.\n"
     "Food stays in the lobby.\nQuiet floor is upstairs.\nReturns go in the slot.")
assert shared_lines(T, N, 100, 30) == [1]
q(1, CHK, "B", "<p>Which line appears in <strong>both</strong> chunk 1 and chunk 2?</p>",
  [("“Printing costs ten cents.”", None),
   ("“Study rooms must be booked.”", "It gets popped. After dropping line 1 the buffer is 53 — still over 30 — so line 2 goes too."),
   ("“Food stays in the lobby.”", "That line never makes it into chunk 1: 81 + 24 + 1 = 106 is over 100, so it starts chunk 2."),
   ("None — the chunks share no line", "The popping stops at 25, which is not over 30, so line 3 survives into chunk 2.")],
  "<p>Atoms: 27, 27, 25, 24, 24, 23.</p>"
  "<p>Buffer: 27 → 55 → 81. Line 4: 81 + 24 + 1 = 106, over 100. <strong>Emit chunk 1 = 81</strong> (lines 1–3).</p>"
  "<p>Pop while over 30: drop line 1 → 81 − 27 − 1 = 53. Still over 30, drop line 2 → 53 − 27 − 1 = 25. Stop. "
  "<strong>Line 3, “Printing costs ten cents.”, carries.</strong></p>"
  "<p>Buffer: 25 → 50 → 75 → 99. End: <strong>emit chunk 2 = 99</strong> (lines 3–6).</p>",
  doc=(T, N, 100, 30, [81, 99]))

T = "Feed the cat.\nFill the bowl.\nSweep the porch.\nFold the laundry."
q(1, EDGE, "A", "<p>After chunk 1 is emitted, does “Fill the bowl.” carry into chunk 2?</p>",
  [("Yes — after popping line 1 the buffer is exactly 14, which is not over 14", None),
   ("No — with its newline it counts as 15, which is over 14",
    "A single atom in the buffer has no separator. Popping line 1 takes the newline between lines 1 and 2 with it: 28 − 13 − 1 = 14."),
   ("No — overlap carries at most 14 characters, never a whole line",
    "Atoms are never split. Overlap carries whole lines or nothing."),
   ("Yes — but “Sweep the porch.” then won't fit beside it, so it gets popped anyway",
    "Check it: 14 + 1 + 16 = 31, which is not over 32. It fits, so nothing more is popped.")],
  "<p>Atoms: 13, 14, 16, 17.</p>"
  "<p>Buffer 13 → 13 + 14 + 1 = 28. Line 3: 28 + 16 + 1 = 45, over 32. <strong>Emit chunk 1 = 28.</strong></p>"
  "<p>Pop while over 14: drop line 1 <em>and the newline that joined it to line 2</em> → 28 − 13 − 1 = 14. "
  "14 is not over 14, so stop. Does line 3 fit beside it? 14 + 1 + 16 = 31, not over 32 — yes.</p>"
  "<p><strong>Yes, it carries.</strong> The chunks are 28, 31, 17.</p>",
  doc=(T, N, 32, 14, [28, 31, 17]))

T = "Field trip Friday.\nPack a lunch.\nThe bus will leave the school parking lot at seven sharp."
q(1, EDGE, "C", "<p>Give the chunks.</p>",
  [("32 and 57, with no line shared", None),
   ("32 and 71, sharing “Pack a lunch.”",
    "71 is over the Chunk Size of 60. The popping also continues while the next atom won't fit beside what's left: 13 + 1 + 57 = 71, over 60 — so “Pack a lunch.” has to go."),
   ("Three chunks: 18, 13 and 57", "Lines 1 and 2 fit together: 18 + 13 + 1 = 32, not over 60."),
   ("32 and 60 — the long line is trimmed to fit", "Atoms are never split. The 57-character line goes in whole.")],
  "<p>Atoms: 18, 13, 57.</p>"
  "<p>Buffer 18 → 32. Line 3: 32 + 57 + 1 = 90, over 60. <strong>Emit chunk 1 = 32.</strong></p>"
  "<p>Pop while over 15: drop line 1 → 13. The overlap test is satisfied… but line 3 still has to fit beside it: "
  "13 + 1 + 57 = 71, over 60. So pop again → 0.</p>"
  "<p>Buffer takes line 3: 57. <strong>Chunks 32 and 57, nothing shared.</strong> Chunk Size wins over overlap.</p>",
  doc=(T, N, 60, 15, [32, 57]))

PR_A = ("<p>A search engine runs over <strong>20 sentences</strong> from a bike-repair guide. The question is "
        "<em>“How do I fix a flat tire?”</em> The answer key marks <strong>5 sentences relevant</strong> "
        "(and 15 not relevant). You set <strong>k = 4</strong>, and <strong>3</strong> of the 4 results are relevant.</p>")
assert (pct(3, 4), pct(3, 5), pct(3, 20), pct(1, 4), pct(3, 15)) == (75, 60, 15, 25, 20)
q(1, PR, "D", PR_A + "<p>What is the <strong>precision</strong>?</p>",
  [("75%", None),
   ("60%", "That is 3 ÷ 5 — dividing by the relevant sentences. That's recall."),
   ("15%", "That is 3 ÷ 20 — dividing by the whole collection. Precision only asks about what came back."),
   ("25%", "That is 1 ÷ 4 — the share of the results that was junk, the opposite of precision.")],
  "<p>Precision = hits ÷ k = 3 ÷ 4 = <strong>75%</strong>. Of the 4 results, 3 were relevant.</p>")
q(1, PR, "B", PR_A + "<p>What is the <strong>recall</strong>?</p>",
  [("60%", None),
   ("75%", "That is 3 ÷ 4 — dividing by k. That's precision."),
   ("15%", "That is 3 ÷ 20 — dividing by every sentence in the collection."),
   ("20%", "That is 3 ÷ 15 — dividing by the not-relevant sentences, which aren't in either formula.")],
  "<p>Recall = hits ÷ relevant = 3 ÷ 5 = <strong>60%</strong>. Of the 5 sentences that could have helped, the search found 3.</p>")

# ================================================================== TEST 2
u, w = (1, 3), (2, -2)
assert (2 * u[0] - w[0], 2 * u[1] - w[1]) == (0, 8)
q(2, VEC, "C", "<p>Let <b>u</b> = (1, 3) and <b>v</b> = (2, −2). Compute 2<b>u</b> − <b>v</b>.</p>",
  [(v(0, 8), None),
   (v(4, 4), "That is 2<b>u</b> + <b>v</b>. Subtracting (2, −2) means subtracting 2 and subtracting −2."),
   (v(0, 4), "A sign slip in the second component: 6 − (−2) = 6 + 2 = 8, not 4."),
   (v(-1, 5), "That is <b>u</b> − <b>v</b> — the 2 never got applied.")],
  "<p>Scale first: 2<b>u</b> = (2, 6).</p><p>Then subtract: (2 − 2, 6 − (−2)) = <strong>(0, 8)</strong>.</p>")

vs = [(4, 1), (-2, 3), (1, 5)]
assert tuple(F(sum(c), 3) for c in zip(*vs)) == (1, 3)
q(2, VEC, "D", "<p>A three-word phrase has (2-D) word embeddings (4, 1), (−2, 3) and (1, 5). Mean-pool them into one phrase vector.</p>",
  [(v(1, 3), None),
   (v(3, 9), "That is the sum. Mean pooling then divides by the number of words, 3."),
   (v(1.5, 4.5), "Divided by 2 — but there are three vectors."),
   (f"({fr(7, 3)}, 3)", "A sign slip: 4 + (−2) + 1 = 3, not 7.")],
  "<p>Add them: (4 + (−2) + 1, 1 + 3 + 5) = (3, 9).</p><p>Divide by 3: <strong>(1, 3)</strong>.</p>")

assert dot((2, 4), (6, -3)) == 0
q(2, DOT, "A", "<p>For what value of <em>k</em> is (2, <em>k</em>) perpendicular to (6, −3)?</p>",
  [("<em>k</em> = 4", None),
   ("<em>k</em> = −4", "Check it: (2)(6) + (−4)(−3) = 12 + 12 = 24, not 0. Watch the sign on −3."),
   ("<em>k</em> = 9", "That is 12 − 3 — the <em>k</em> fell out of the equation."),
   (f"<em>k</em> = {fr(1, 4)}", f"Solved 3<em>k</em> = 12 upside down: <em>k</em> = 12 ÷ 3, not 3 ÷ 12.")],
  "<p>Perpendicular means the dot product is 0: (2)(6) + (<em>k</em>)(−3) = 12 − 3<em>k</em> = 0.</p>"
  "<p>So 3<em>k</em> = 12 and <strong><em>k</em> = 4</strong>. Check: (2, 4) · (6, −3) = 12 − 12 = 0. ✓</p>")

pairs = [((1, -2), (4, 2)), ((3, 1), (1, -2)), ((-2, 3), (4, 1)), ((2, 2), (1, 3))]
assert [dot(a, b) for a, b in pairs] == [0, 1, -5, 8]
q(2, DOT, "C", "<p>Using only the <em>sign</em> of the dot product, which pair points in broadly <strong>opposite</strong> directions?</p>",
  [("(−2, 3) and (4, 1)", None),
   ("(1, −2) and (4, 2)", "(1)(4) + (−2)(2) = 4 − 4 = 0. Zero means perpendicular, not opposite."),
   ("(3, 1) and (1, −2)", "(3)(1) + (1)(−2) = 3 − 2 = 1. Positive: same general direction."),
   ("(2, 2) and (1, 3)", "(2)(1) + (2)(3) = 2 + 6 = 8. Positive: same general direction.")],
  "<p>(−2)(4) + (3)(1) = −8 + 3 = <strong>−5</strong>. A negative dot product means broadly opposite directions.</p>"
  "<p>The other pairs give 0 (perpendicular), 1 and 8 (both same general direction).</p>")

T = "Charge the robot.\nCheck the wheels.\nLoad the program.\nTest the sensor."
q(2, CHK, "B", "<p>Give the chunks and their lengths.</p>",
  [("Three chunks: 17, 17, 34", None),
   ("Four chunks: 17, 17, 17, 16", "The last two lines do fit together: 17 + 16 + 1 = 34, and 34 is not <em>over</em> 34."),
   ("Two chunks: 34 and 33", "Forgot the separator. 17 + 17 + 1 = 35 is over 34, so the first two lines can't share a chunk."),
   ("Two chunks: 35 and 34", "A 35-character chunk is over the Chunk Size — lines 1 and 2 don't fit together.")],
  "<p>Atoms: 17, 17, 17, 16.</p>"
  "<p>Buffer 17. Line 2: 17 + 17 + 1 = 35, over 34. <strong>Emit 17.</strong> Line 3: same thing, <strong>emit 17.</strong></p>"
  "<p>Buffer 17 (line 3). Line 4: 17 + 16 + 1 = <strong>34</strong> — not <em>over</em> 34, so it fits. End: <strong>emit 34.</strong></p>"
  "<p><strong>Three chunks: 17, 17, 34.</strong> Landing exactly on the Chunk Size is allowed.</p>",
  doc=(T, N, 34, 0, [17, 17, 34]))

T = ("The science fair is held in the gym.\nJudges arrive at nine in the morning.\n"
     "Posters must be taped to the tables.")
assert shared_lines(T, N, 80, 20) == [0]
q(2, CHK, "A", "<p>How many lines does chunk 2 share with chunk 1?</p>",
  [("None", None),
   ("One — “Judges arrive at nine in the morning.”", "That line is 37 characters — over the overlap budget of 20 — so it gets popped too."),
   ("About 20 characters of “Judges arrive at nine…”", "Atoms are never split. A line carries whole or not at all."),
   ("Two — both lines of chunk 1", "Popping continues while the buffer is over 20, and each line alone is over 20.")],
  "<p>Atoms: 36, 37, 36.</p>"
  "<p>Buffer 36 → 36 + 37 + 1 = 74. Line 3: 74 + 36 + 1 = 111, over 80. <strong>Emit chunk 1 = 74.</strong></p>"
  "<p>Pop while over 20: drop line 1 → 37, still over 20; drop line 2 → 0. Buffer takes line 3. <strong>Emit chunk 2 = 36.</strong></p>"
  "<p><strong>Zero overlap</strong>, even with Chunk Overlap set to 20: every line is bigger than 20, and you can't keep part of a line.</p>",
  doc=(T, N, 80, 20, [74, 36]))

T = "Welcome to robotics club.\nBring tools\nCharge bots\nThe competition is next Saturday."
assert shared_lines(T, N, 60, 22) == [1]
q(2, EDGE, "B", "<p>The two short lines are 11 characters each, and Chunk Overlap is 22. How many of them carry into chunk 2?</p>",
  [("One — “Charge bots”", None),
   ("Both — 11 + 11 = 22, which is not over 22",
    "Two lines in the buffer have a newline between them: 11 + 1 + 11 = 23, which <em>is</em> over 22."),
   ("None", "After dropping “Bring tools” the buffer is 11, not over 22 — and 11 + 1 + 33 = 45 fits. So “Charge bots” stays."),
   ("One — “Bring tools”", "Popping always removes from the <em>front</em>, so the earlier line, “Bring tools”, is the one that goes.")],
  "<p>Atoms: 25, 11, 11, 33.</p>"
  "<p>Buffer 25 → 37 → 49. Line 4: 49 + 33 + 1 = 83, over 60. <strong>Emit chunk 1 = 49.</strong></p>"
  "<p>Pop while over 22: drop line 1 → 49 − 25 − 1 = 23. That's 11 + 1 + 11 — the newline between the short lines counts — "
  "so it is still over 22. Drop “Bring tools” → 11. Stop. Line 4 fits: 11 + 1 + 33 = 45.</p>"
  "<p><strong>Only one line, “Charge bots”, carries.</strong> Chunks: 49 and 45.</p>",
  doc=(T, N, 60, 22, [49, 45]))

T = "Art is Monday.\n\nBring a smock.\n\nPaint the mural.\n\nWash brushes."
q(2, EDGE, "D", "<p>Give the chunks. (Careful: what is the separator here?)</p>",
  [("48 and 31", None),
   ("46 and 30", "Counted each separator as 1 character. <code>\\n\\n</code> is two characters, so every join costs 2."),
   ("48 and 47", "Stopped popping at 32 — but 32 is over 30, so “Bring a smock.” has to go too."),
   ("44 and 43", "Left the separators out altogether.")],
  "<p>Atoms: 14, 14, 16, 13. The blank lines are the separators, and each is <strong>2</strong> characters.</p>"
  "<p>Buffer 14 → 14 + 14 + 2 = 30 → 30 + 16 + 2 = 48. Atom 4: 48 + 13 + 2 = 63, over 50. <strong>Emit chunk 1 = 48.</strong></p>"
  "<p>Pop while over 30: drop atom 1 → 48 − 14 − 2 = 32, still over 30; drop atom 2 → 16. Stop.</p>"
  "<p>Add atom 4: 16 + 2 + 13 = 31. <strong>Chunks 48 and 31</strong>, sharing “Paint the mural.”</p>",
  doc=(T, NN, 50, 30, [48, 31]))

PR_B = ("<p>A search engine runs over <strong>40 sentences</strong> from a campus FAQ. The question is "
        "<em>“When is the library open?”</em> The answer key marks <strong>4 sentences relevant</strong> "
        "(and 36 not relevant). You set <strong>k = 10</strong>, and <strong>2</strong> of the 10 results are relevant.</p>")
assert (pct(min(10, 4), 10), pct(2, 10), pct(4, 40), pct(2, 4), pct(2, 40), pct(10, 40)) == (40, 20, 10, 50, 5, 25)
q(2, PR, "C", PR_B + "<p>What was the <strong>best possible precision</strong> at this k?</p>",
  [("40%", None),
   ("100%", "Assumes a perfect score is always reachable. Only 4 relevant sentences exist, so a perfect search still fills 6 of the 10 slots with junk."),
   ("20%", "That is 2 ÷ 10 — the precision this search <em>actually</em> got, not the best it could get."),
   ("10%", "That is 4 ÷ 40 — relevant over the whole collection.")],
  "<p>Best possible hits = the smaller of k (10) and relevant (4) = 4.</p>"
  "<p>Best possible precision = 4 ÷ 10 = <strong>40%</strong>.</p>")
q(2, PR, "A", PR_B + "<p>What is the <strong>recall</strong>?</p>",
  [("50%", None),
   ("20%", "That is 2 ÷ 10 — dividing by k. That's precision."),
   ("5%", "That is 2 ÷ 40 — dividing by the whole collection."),
   ("100%", "That's the <em>best possible</em> recall at k = 10 (4 ÷ 4), not what this search got.")],
  "<p>Recall = hits ÷ relevant = 2 ÷ 4 = <strong>50%</strong>. Half of the relevant sentences were found.</p>")

# ================================================================== TEST 3
assert math.isqrt(norm2((2, -3, 6))) == 7
q(3, VEC, "D", "<p>Find the length of the 3-D vector (2, −3, 6).</p>",
  [("7", None),
   ("11", "That is 2 + 3 + 6 — adding the sizes of the components instead of squaring them."),
   ("5", "That is 2 + (−3) + 6 — adding the components with their signs."),
   ("49", "That is 4 + 9 + 36 — the right sum, but the square root is missing.")],
  "<p>Same rule as 2-D, one more squared term: √(2² + (−3)² + 6²) = √(4 + 9 + 36) = √49 = <strong>7</strong>.</p>")

q(3, VEC, "A", "<p>Suppose ‖<b>v</b>‖ = 6. <em>Without</em> knowing the components of <b>v</b>, find ‖−3<b>v</b>‖.</p>",
  [("18", None),
   ("−18", "A length is never negative. The rule is ‖c<b>v</b>‖ = |c| ‖<b>v</b>‖."),
   ("2", "Divided by 3 instead of multiplying."),
   ("54", "Multiplied by 3² = 9. Every component gets multiplied by −3, so the length gets multiplied by |−3| = 3.")],
  "<p>‖c<b>v</b>‖ = |c| ‖<b>v</b>‖, so ‖−3<b>v</b>‖ = |−3| × 6 = <strong>18</strong>.</p>"
  "<p>The minus sign flips the direction but doesn't change the length.</p>")

assert dot((2, -1, 3), (1, 4, 2)) == 4
q(3, DOT, "B", "<p>Compute the 3-D dot product (2, −1, 3) · (1, 4, 2).</p>",
  [("4", None),
   ("12", "A sign slip: (−1)(4) = −4, so it's 2 − 4 + 6."),
   ("11", "Added all six numbers together. Multiply matching components first, then add."),
   ("(2, −4, 6)", "Multiplied matching components but never added them. A dot product is one number.")],
  "<p>(2)(1) + (−1)(4) + (3)(2) = 2 − 4 + 6 = <strong>4</strong>.</p>")

q(3, DOT, "A", "<p>A vector <b>u</b> has <b>u</b> · <b>u</b> = 49. What is ‖<b>u</b>‖?</p>",
  [("7", None),
   ("49", "A vector dotted with itself is its length <em>squared</em>. Take the square root."),
   ("24.5", "Halved instead of taking the square root."),
   ("−7", "Lengths are never negative.")],
  "<p><b>u</b> · <b>u</b> = ‖<b>u</b>‖², so ‖<b>u</b>‖ = √49 = <strong>7</strong>.</p>")

T = ("Lab rules.\n\nEvery student must wear goggles whenever a burner is lit, "
     "and long hair must be tied back.")
q(3, CHK, "C", "<p>Give the chunks.</p>",
  [("Two chunks: 10 and 90 — the second is longer than the Chunk Size", None),
   ("Three chunks: 10, 60 and 30", "Split Text never cuts an atom. The whole paragraph is one atom, so it goes into a chunk in one piece."),
   ("Two chunks: 10 and 60 — the rest of the paragraph is dropped", "Nothing is ever thrown away."),
   ("No chunks — Split Text reports an error", "It doesn't error. It emits the oversized chunk and only writes a warning to its log.")],
  "<p>The separator is <code>\\n\\n</code>, so the blank line is the cut. Atoms: 10 and 90.</p>"
  "<p>Buffer 10. Atom 2: 10 + 90 + 2 = 102, over 60. <strong>Emit chunk 1 = 10.</strong> Pop → empty. "
  "Buffer takes atom 2: 90. End: <strong>emit chunk 2 = 90.</strong></p>"
  "<p>That's 30 over the Chunk Size, and Split Text lets it happen. Lowering Chunk Size wouldn't help — "
  "only a finer <strong>Separator</strong> can break up a paragraph.</p>",
  doc=(T, NN, 60, 5, [10, 90]))

T = "cat\ndog\nbird\nfish\nfrog\nbee"
assert merge(T, N, 12, 0)[0] == "cat\ndog\nbird"
q(3, CHK, "C", "<p>A word list, one word per line. What is chunk 1?</p>",
  [("cat, dog and bird together", None),
   ("just cat", "Chunk Size 12 isn't “one word”. The buffer keeps gluing while the next word fits."),
   ("cat and dog together", "Adding bird: 7 + 4 + 1 = 12, and 12 is not <em>over</em> 12 — it fits."),
   ("cat, dog, bird and fish together", "Adding fish: 12 + 4 + 1 = 17, over 12.")],
  "<p>Atoms: 3, 3, 4, 4, 4, 3.</p>"
  "<p>Buffer 3 → 3 + 3 + 1 = 7 → 7 + 4 + 1 = <strong>12</strong> (exactly the Chunk Size, which is allowed). "
  "Fish: 12 + 4 + 1 = 17, over 12. <strong>Emit chunk 1 = “cat, dog, bird”</strong> (12 characters).</p>"
  "<p>The whole run gives three chunks: cat/dog/bird, fish/frog, bee.</p>",
  doc=(T, N, 12, 0, [12, 9, 3]))

T = "Soccer practice.\nBring shin guards.\nFill a bottle.\nThe coach will run sprint drills at the end."
assert shared_lines(T, N, 60, 35) == [1]
q(3, EDGE, "D", "<p>How many lines does chunk 2 share with chunk 1?</p>",
  [("One — “Fill a bottle.”", None),
   ("Two — “Bring shin guards.” and “Fill a bottle.”",
    "After dropping line 1 the buffer is 33, which passes the overlap test — but 33 + 1 + 44 = 78 is over 60, so the popping keeps going."),
   ("None", "After dropping line 2 the buffer is 14, and 14 + 1 + 44 = 59 fits. So “Fill a bottle.” stays."),
   ("Three — all of chunk 1", "Chunk 1 is 50 characters, over the overlap of 35; at least line 1 has to go.")],
  "<p>Atoms: 16, 18, 14, 44.</p>"
  "<p>Buffer 16 → 35 → 50. Line 4: 50 + 44 + 1 = 95, over 60. <strong>Emit chunk 1 = 50.</strong></p>"
  "<p>Pop: drop line 1 → 33. That's not over 35, <em>but</em> line 4 has to fit beside it: 33 + 1 + 44 = 78, over 60. "
  "Pop again → 14. Now 14 + 1 + 44 = 59, which fits. Stop.</p>"
  "<p><strong>One line carries.</strong> Chunks: 50 and 59. The overlap allowed two lines; the Chunk Size only had room for one.</p>",
  doc=(T, N, 60, 35, [50, 59]))

T = "Mix the batter.\nGrease the pan.\nBake it for thirty minutes at three fifty."
q(3, EDGE, "B", "<p>How long is chunk 2?</p>",
  [("58", None),
   ("42", "Popped “Grease the pan.” because 15 + 1 + 42 = 58 — but 58 is not <em>over</em> 58, so it fits and stays."),
   ("57", "Forgot the newline that joins the carried line to the new one."),
   ("31", "That's chunk 1.")],
  "<p>Atoms: 15, 15, 42.</p>"
  "<p>Buffer 15 → 31. Line 3: 31 + 42 + 1 = 74, over 58. <strong>Emit chunk 1 = 31.</strong></p>"
  "<p>Pop while over 15: drop line 1 → 15. Stop. Does line 3 fit beside it? 15 + 1 + 42 = <strong>58</strong>, not over 58 — yes.</p>"
  "<p><strong>Chunk 2 = 58.</strong> Both checks in the popping are strictly “bigger than”.</p>",
  doc=(T, N, 58, 15, [31, 58]))

assert 10 * F(40, 100) == 4 and 4 / F(80, 100) == 5
q(3, PR, "B", "<p>Your search returns <strong>k = 10</strong> results. Precision is <strong>40%</strong> and recall is "
              "<strong>80%</strong>. How many sentences does the answer key mark as relevant?</p>",
  [("5", None),
   ("4", "That's the number of hits — one step early."),
   ("8", "That's 80% of 10 — using k where hits belongs."),
   ("25", "That's 10 ÷ 40%, which doesn't mean anything here.")],
  "<p>Precision = hits ÷ k, so 40% = hits ÷ 10, and hits = 4.</p>"
  "<p>Recall = hits ÷ relevant, so 80% = 4 ÷ relevant. 4 is 80% of what? <strong>5.</strong></p>")

q(3, PR, "D", "<p>A question has <strong>6 relevant</strong> sentences in a collection of <strong>30</strong>. "
              "Which k lets a perfect search score 100% on <strong>both</strong> precision and recall?</p>",
  [("k = 6", None),
   ("k = 1", "Best precision is 1 ÷ 1 = 100%, but best recall is only 1 ÷ 6."),
   ("k = 10", "Best recall is 100%, but best precision is only 6 ÷ 10 = 60%."),
   ("k = 30", "Returning everything guarantees 100% recall, but best precision is only 6 ÷ 30 = 20%.")],
  "<p>When k equals the number of relevant sentences, best possible hits = 6, so best precision = 6 ÷ 6 and best recall = 6 ÷ 6. "
  "<strong>k = 6</strong> is the only setting where both can reach 100%.</p>")

# ================================================================== TEST 4
assert math.isqrt(norm2((-5, 12))) == 13
q(4, VEC, "A", "<p>Find the unit vector pointing the same way as (−5, 12).</p>",
  [(f"(−{fr(5, 13)}, {fr(12, 13)})", None),
   (f"(−{fr(5, 7)}, {fr(12, 7)})", "Divided by −5 + 12 = 7. Divide by the <em>length</em>."),
   (f"({fr(5, 13)}, {fr(12, 13)})", "Dropped the minus sign — that vector points a different way."),
   (f"(−{fr(5, 17)}, {fr(12, 17)})", "Divided by 5 + 12 = 17. Divide by the length, √(25 + 144).")],
  f"<p>Length: √((−5)² + 12²) = √(25 + 144) = √169 = 13.</p>"
  f"<p>Divide every component by 13: <strong>(−{fr(5, 13)}, {fr(12, 13)})</strong>.</p>")

assert math.isqrt(norm2((5, 12))) == 13
q(4, VEC, "B", "<p>Let <b>u</b> = (5, 0) and <b>v</b> = (0, 12). Find ‖<b>u</b> + <b>v</b>‖.</p>",
  [("13", None),
   ("17", "That's ‖<b>u</b>‖ + ‖<b>v</b>‖ = 5 + 12. The length of a sum is generally <em>not</em> the sum of the lengths."),
   ("169", "That's 25 + 144 — the square root is missing."),
   ("√17", "Added 5 + 12 and then took the root. Square each component first.")],
  "<p><b>u</b> + <b>v</b> = (5, 12), so ‖<b>u</b> + <b>v</b>‖ = √(25 + 144) = √169 = <strong>13</strong>.</p>"
  "<p>Compare 5 + 12 = 17: the straight path is shorter than the detour.</p>")

q(4, DOT, "D", "<p>Two vectors <b>u</b> and <b>v</b> have cosine similarity 0.6. What is the cosine similarity of 5<b>u</b> and −2<b>v</b>?</p>",
  [("−0.6", None),
   ("0.6", "Stretching doesn't change cosine similarity — but the minus sign in −2<b>v</b> flips it to point the opposite way."),
   ("−6", "Multiplied 0.6 by 5 and by −2. Cosine similarity ignores length, and it can never go below −1."),
   ("−1.2", "Multiplied by −2. Cosine similarity ignores length, and it can never go below −1.")],
  "<p>Cosine similarity ignores length, so the 5 and the 2 don't matter. The minus sign reverses the direction of <b>v</b>, "
  "which flips the sign of the score: <strong>−0.6</strong>.</p>")

assert dot((2, 0), (0, -7)) == 0
q(4, DOT, "C", "<p>Find the cosine similarity of (2, 0) and (0, −7).</p>",
  [("0", None),
   ("−1", "A minus sign on a component doesn't make two vectors opposite. The dot product here is 0."),
   ("1", "Check the dot product: (2)(0) + (0)(−7) = 0."),
   ("−14", "That's (2)(−7) — multiplying components that don't match up.")],
  "<p>Dot product: (2)(0) + (0)(−7) = 0. Zero divided by anything is 0, so the cosine similarity is <strong>0</strong>.</p>"
  "<p>One points along the horizontal axis, the other straight down: perpendicular.</p>")

T = "Games start at six.\nBring a snack.\n\nThe gym doors lock.\nUse the side door."
assert [len(c) for c in merge(T, NN, 40, 15)] == [34, 38]
q(4, CHK, "A", "<p>Chunk Size <strong>40</strong> and Chunk Overlap <strong>15</strong> for both runs. "
               "Run it once with Separator <code>\\n\\n</code> and once with Separator <code>\\n</code>. "
               "How many chunks does each run make?</p>",
  [("<code>\\n\\n</code>: 2 chunks · <code>\\n</code>: 3 chunks", None),
   ("<code>\\n\\n</code>: 2 chunks · <code>\\n</code>: 4 chunks", "With <code>\\n</code>, lines get glued: 19 + 14 + 1 = 34 fits under 40. Four atoms doesn't mean four chunks."),
   ("<code>\\n\\n</code>: 1 chunk · <code>\\n</code>: 3 chunks", "The two paragraphs are 34 and 38; together that's 34 + 38 + 2 = 74, over 40."),
   ("Both runs make 2 chunks", "With <code>\\n</code>, a line carries over as overlap, which pushes the last line into a third chunk.")],
  "<p><strong>With <code>\\n\\n</code>:</strong> two atoms, each a whole paragraph with its newline inside: "
  "19 + 1 + 14 = 34 and 19 + 1 + 18 = 38. 34 + 38 + 2 = 74 is over 40, so <strong>2 chunks: 34 and 38</strong>, no overlap.</p>"
  "<p><strong>With <code>\\n</code>:</strong> atoms 19, 14, 19, 18 (the blank line is an empty piece and gets dropped). "
  "19 → 34; next: 34 + 19 + 1 = 54, over 40, emit 34. Pop to 14 (“Bring a snack.” carries). 14 + 19 + 1 = 34; next: 53, over 40, emit 34. "
  "Pop: 34 − 14 − 1 = 19, over 15, pop → 0. Last line: 18. <strong>3 chunks: 34, 34, 18.</strong></p>",
  doc=(T, N, 40, 15, [34, 34, 18]),
  settings="Chunk Size <strong>40</strong> · Chunk Overlap <strong>15</strong> · two runs, two separators")

T = "Plug it in.\nPress start.\nWait a bit.\nPick a song.\nTurn it up.\nDance now.\nRest after."
assert shared_lines(T, N, 50, 24) == [2, 2]
q(4, CHK, "D", "<p>How many chunks come out, and how many lines does chunk 2 share with chunk 1?</p>",
  [("3 chunks; chunk 2 shares 2 lines", None),
   ("2 chunks; no lines shared", "That's what happens with no overlap at all. Chunk Overlap 24 lets lines carry."),
   ("2 chunks; chunk 2 shares 1 line", "After two pops the buffer is exactly 24 — not <em>over</em> 24 — so popping stops there, with two lines."),
   ("7 chunks, one per line", "The separator makes atoms, not chunks. Short lines get glued together.")],
  "<p>Atoms: 11, 12, 11, 12, 11, 10, 11.</p>"
  "<p>Buffer: 11 → 24 → 36 → 49. Line 5: 49 + 11 + 1 = 61, over 50. <strong>Emit chunk 1 = 49.</strong> "
  "Pop while over 24: → 37 → <strong>24</strong>. Stop. Lines 3 and 4 carry.</p>"
  "<p>Buffer: 24 → 36 → 47. Line 7: 47 + 11 + 1 = 59, over 50. <strong>Emit chunk 2 = 47.</strong> Pop: → 35 → 22. Lines 5 and 6 carry. "
  "22 + 11 + 1 = 34. End: <strong>emit chunk 3 = 34.</strong></p>"
  "<p><strong>3 chunks; each shares 2 lines with the one before.</strong></p>",
  doc=(T, N, 50, 24, [49, 47, 34]))

T = "Check in at the gate.\nShow your ticket.\nThe parade starts at the corner of Main and Fifth."
q(4, EDGE, "C", "<p>A classmate's work: <em>“Atoms 21, 17, 50. Chunk 1 = 39. Pop while over 17: drop line 1, leaving 17. "
                "Stop, so ‘Show your ticket.’ carries. Add line 3: 17 + 1 + 50 = 68. Chunks: 39 and 68.”</em></p>"
                "<p>What went wrong?</p>",
  [("They stopped popping too soon: 17 + 1 + 50 = 68 is over 60, so “Show your ticket.” must go too. Chunks: 39 and 50.", None),
   ("Nothing — 17 is not over 17, so the line carries", "The overlap test passes, but the popping also checks that the next atom fits: 68 is over 60."),
   ("They forgot a newline when popping: the buffer is really 18, so the line goes. Chunks: 39 and 50.",
    "Right chunks, wrong reason. Popping line 1 takes its newline with it — the buffer really is 17."),
   ("Chunk 1 is wrong: it should be 38", "Two lines joined need one newline: 21 + 17 + 1 = 39.")],
  "<p>The warning sign is in their own answer: a 68-character chunk with Chunk Size 60, when no single line is anywhere near 60.</p>"
  "<p>After popping to 17, the overlap test is satisfied, but line 3 has to fit beside it: 17 + 1 + 50 = 68, over 60. "
  "So “Show your ticket.” is dropped too. <strong>Chunks: 39 and 50, nothing shared.</strong></p>",
  doc=(T, N, 60, 17, [39, 50]))

T = "Grab a tray.\nTake a fork.\nPick a drink.\nPay at the register by the door."
q(4, EDGE, "B", "<p>Chunk 1 is the first three lines, 39 characters. Popping begins. Right after “Grab a tray.” is dropped, how big is the buffer?</p>",
  [("26", None),
   ("25", "Left out the newline between the two lines still in the buffer: 12 + 1 + 13 = 26."),
   ("27", "39 − 12 forgets that the newline after “Grab a tray.” leaves with it: 39 − 12 − 1 = 26."),
   ("13", "That's after a <em>second</em> pop. The question asks about right after the first.")],
  "<p>Atoms: 12, 12, 13, 32. Chunk 1 = 12 + 12 + 13 + 2 newlines = 39.</p>"
  "<p>Dropping “Grab a tray.” removes 12 characters <em>and</em> the newline that joined it to line 2: 39 − 12 − 1 = <strong>26</strong>. "
  "(Check: what's left is “Take a fork.” + newline + “Pick a drink.” = 12 + 1 + 13 = 26.)</p>"
  "<p>For the record: 26 is over 25, so popping continues; the final chunks are 39 and 32.</p>",
  doc=(T, N, 45, 25, [39, 32]))

q(4, PR, "A", "<p>An answer key marks <strong>6 sentences relevant</strong> and <strong>4 not relevant</strong> (10 in all). "
              "Which of these results <strong>could not</strong> happen?</p>",
  [("Return 8 results and get 3 hits", None),
   ("Return 4 results and get 0 hits", "Possible: all 4 not-relevant sentences come back. A terrible search, but a possible one."),
   ("Return 10 results and get 6 hits", "Possible — in fact certain. Returning everything returns all 6 relevant sentences."),
   ("Return 3 results and get 3 hits", "Possible: 6 relevant sentences exist, so 3 of them can all come back.")],
  "<p>8 results with 3 hits means 5 results were not relevant — but only 4 not-relevant sentences exist. "
  "Return 8 and at least 8 − 4 = 4 must be relevant.</p>"
  "<p>That's the one job of the not-relevant count: it's in neither formula, but it tells you which results are possible at all.</p>")

PR_C = ("<p>A search engine runs over <strong>25 sentences</strong>. The answer key marks <strong>10 relevant</strong> "
        "(15 not). You set <strong>k = 5</strong>, and <strong>3</strong> of the 5 results are relevant.</p>")
assert (pct(min(5, 10), 10), pct(3, 10), pct(5, 25)) == (50, 30, 20)
q(4, PR, "C", PR_C + "<p>What was the <strong>best possible recall</strong> at this k?</p>",
  [("50%", None),
   ("100%", "With only 5 slots, at least 5 of the 10 relevant sentences are always left out."),
   ("30%", "That's 3 ÷ 10 — the recall this search actually got, not the best it could get."),
   ("20%", "That's 5 ÷ 25 — k over the whole collection.")],
  "<p>Best possible hits = the smaller of k (5) and relevant (10) = 5.</p>"
  "<p>Best possible recall = 5 ÷ 10 = <strong>50%</strong>. This search got 30% against a ceiling of 50%.</p>")

# ================================================================== TEST 5
assert norm2((-2, 4)) == 20
q(5, VEC, "C", "<p>Find ‖(−2, 4)‖. Leave it as a square root if it isn't a whole number.</p>",
  [("√20 (which is also 2√5)", None),
   ("6", "That's 2 + 4 — adding the sizes of the components."),
   ("√12", "(−2)² is +4, not −4. The slip gives 16 − 4 = 12."),
   ("20", "The right sum, but the square root is missing.")],
  "<p>√((−2)² + 4²) = √(4 + 16) = <strong>√20</strong>. It isn't a whole number, so leave it as a root "
  "(it simplifies to 2√5, since 20 = 4 × 5).</p>")

assert F(36, 100) + F(64, 100) == 1
q(5, VEC, "D", "<p>Which of these is a <strong>unit vector</strong> (length exactly 1)?</p>",
  [("(0.6, −0.8)", None),
   ("(0.5, 0.5)", "Length √(0.25 + 0.25) = √0.5, which is less than 1."),
   ("(1, 1)", "Length √(1 + 1) = √2, more than 1."),
   ("(0.3, 0.7)", "The components add to 1 — that's what a <em>probability distribution</em> needs, not a unit vector. "
                  "Length √(0.09 + 0.49) = √0.58, less than 1.")],
  "<p>‖(0.6, −0.8)‖ = √(0.36 + 0.64) = √1 = <strong>1</strong>.</p>"
  "<p>A unit vector's squared components add to 1. A distribution's plain components add to 1. Different rules.</p>")

A = (F(8, 10), F(6, 10))
cands = [(1, 0), (F(6, 10), F(8, 10)), (0, 1), (-F(8, 10), -F(6, 10))]
assert [dot(A, c) for c in cands] == [F(8, 10), F(96, 100), F(6, 10), -1]
q(5, DOT, "B", "<p>Every word vector below has length 1. Word A = (0.8, 0.6). Which word is <strong>most similar</strong> to A?</p>",
  [("(0.6, 0.8)", None),
   ("(1, 0)", "Similarity (0.8)(1) + (0.6)(0) = 0.8. Close, but 0.96 beats it."),
   ("(0, 1)", "Similarity (0.8)(0) + (0.6)(1) = 0.6."),
   ("(−0.8, −0.6)", "Similarity −0.64 − 0.36 = −1: the exact opposite direction.")],
  "<p>They're all unit vectors, so cosine similarity is just the dot product — no dividing.</p>"
  "<p>(0.8)(0.6) + (0.6)(0.8) = 0.48 + 0.48 = <strong>0.96</strong>, the biggest of the four (the others give 0.8, 0.6 and −1).</p>")

assert dot((1, 2, 2), (2, 1, 2)) == 8 and norm2((1, 2, 2)) == 9
q(5, DOT, "A", "<p>Find the cosine similarity of (1, 2, 2) and (2, 1, 2).</p>",
  [(fr(8, 9), None),
   ("8", "That's the dot product. Now divide by both lengths."),
   (fr(4, 3), "That's 8 ÷ (3 + 3). Multiply the lengths, don't add them. (And a cosine similarity can never be bigger than 1.)"),
   ("1", "Same numbers in a different order — different directions, so the score is below 1.")],
  f"<p>Dot product: (1)(2) + (2)(1) + (2)(2) = 2 + 2 + 4 = 8.</p>"
  f"<p>Lengths: √(1 + 4 + 4) = 3 and √(4 + 1 + 4) = 3.</p>"
  f"<p>Cosine similarity = {fr(8, '3 × 3')} = <strong>{fr(8, 9)}</strong>.</p>")

q(5, CHK, "B", "<p>A classmate builds the Lab 6 word-search flow with Separator <code>\\n</code>, Chunk Size <strong>1000</strong> "
               "and Chunk Overlap <strong>0</strong>, on a file with one word per line. Every search returns one giant chunk "
               "holding every word. Which <strong>one</strong> change fixes it?</p>",
  [("Set Chunk Size to 1", None),
   ("Set Chunk Overlap to 1", "Overlap only controls what carries between chunks. With one chunk, there's nothing to carry."),
   ("Empty out the Separator box", "With no separator, the splitter cuts between every character — and at Chunk Size 1000 it glues them right back into one blob."),
   ("Set the Separator to <code>\\n\\n</code>", "A one-word-per-line file has no blank lines, so the whole file becomes a single atom.")],
  "<p>The cut is fine: <code>\\n</code> gives one atom per word. The problem is the glue: a short word list is nowhere near "
  "1000 characters, so every word fits in the same buffer and one chunk comes out.</p>"
  "<p><strong>Chunk Size 1</strong> means no two words can ever fit together, so every word becomes its own chunk — "
  "and since atoms are never split, you get whole words, not letters.</p>")

T = "Arrive early.\nFind your team.\nWarm up.\nPlay four quarters.\nShake hands.\nClean the bench."
assert [len(c) for c in merge(T, N, 45, 0)] == [38, 32, 16]
q(5, CHK, "D", "<p>How long is chunk 2?</p>",
  [("41", None),
   ("32", "That's what happens with no overlap. Here “Warm up.” carries into chunk 2."),
   ("45", "Chunk Size is a ceiling you bump into, not a length you hit."),
   ("39", "Left out the newlines: 8 + 19 + 12 = 39, but the three lines need two newlines between them.")],
  "<p>Atoms: 13, 15, 8, 19, 12, 16.</p>"
  "<p>Buffer: 13 → 29 → 38. Line 4: 38 + 19 + 1 = 58, over 45. <strong>Emit chunk 1 = 38.</strong></p>"
  "<p>Pop while over 20: → 24 → 8. Stop. “Warm up.” carries.</p>"
  "<p>Buffer: 8 → 8 + 19 + 1 = 28 → 28 + 12 + 1 = 41. Line 6: 41 + 16 + 1 = 58, over 45. <strong>Emit chunk 2 = 41.</strong></p>",
  doc=(T, N, 45, 20, [38, 41, 29]))

T = "Band camp.\n\nBring a stand.\n\nMarching drills run all afternoon on the practice field."
assert [len(c) for c in merge(T, NN, 71, 20)] == [26, 56]
q(5, EDGE, "A", "<p>How many paragraphs does chunk 2 share with chunk 1?</p>",
  [("None — chunk 2 is just the long paragraph", None),
   ("One — “Bring a stand.”, making chunk 2 exactly 71",
    "The separator is <code>\\n\\n</code>, two characters: 14 + 2 + 56 = 72, over 71. With a 1-character separator it would have fit."),
   ("One — “Band camp.”", "Popping always starts at the front, so “Band camp.” is the first to go."),
   ("Two — both short paragraphs", "Chunk 1 is 26, over the overlap of 20, so at least one paragraph has to go.")],
  "<p>Atoms: 10, 14, 56.</p>"
  "<p>Buffer 10 → 10 + 14 + 2 = 26. Atom 3: 26 + 56 + 2 = 84, over 71. <strong>Emit chunk 1 = 26.</strong></p>"
  "<p>Pop while over 20: drop “Band camp.” → 26 − 10 − 2 = 14. The overlap test is satisfied, but atom 3 has to fit: "
  "14 + 2 + 56 = <strong>72</strong>, over 71. Pop again → 0.</p>"
  "<p><strong>Nothing shared.</strong> Chunks: 26 and 56. The 2-character separator made the difference.</p>",
  doc=(T, NN, 71, 20, [26, 56]))

q(5, EDGE, "C", "<p>A classmate says Split Text gave them chunks of <strong>40</strong> and <strong>62</strong> characters with "
                "Chunk Size <strong>50</strong>. No single line in their document is longer than 30 characters. What should you conclude?</p>",
  [("They made a mistake — a chunk can only go over Chunk Size when a single atom is over it", None),
   ("Nothing's wrong — Split Text often runs a little over when overlap is turned on",
    "The popping keeps going until the next atom fits, so overlap can never push a chunk over the size."),
   ("Their Chunk Overlap must be set higher than 50", "Overlap never overrides Chunk Size — Chunk Size wins."),
   ("Nothing's wrong — the last chunk is allowed to run over", "The last chunk follows the same rules as every other chunk.")],
  "<p>Chunk Size wins: after each emit, popping continues while the next atom won't fit. So the only way a chunk can be "
  "over the size is if one atom is over the size by itself — and here no line is over 30.</p>"
  "<p>If your chunk is over the size and no atom is, <strong>recheck your popping</strong>.</p>")

q(5, PR, "D", "<p>A question has <strong>5 relevant</strong> sentences, and you set <strong>k = 5</strong>. "
              "The search gets <strong>4</strong> hits. What are the precision and recall?</p>",
  [("Precision 80%, recall 80%", None),
   ("Precision 80%, recall 100%", "Recall is 4 ÷ 5 — one relevant sentence was missed."),
   ("Precision 100%, recall 80%", "Precision is 4 ÷ 5 — one of the five results was junk."),
   ("Precision 100%, recall 100%", "100% on both is the <em>best possible</em> at this k. This search got 4, not 5.")],
  "<p>Precision = 4 ÷ 5 = 80%. Recall = 4 ÷ 5 = 80%.</p>"
  "<p>When k equals the number of relevant sentences, the two fractions have the same top <em>and</em> the same bottom, "
  "so <strong>precision and recall are always equal</strong>.</p>")

assert (pct(4, 8), pct(3, 5), pct(3, 4), pct(4, 5)) == (50, 60, 75, 80)
q(5, PR, "B", "<p>Two students search the same 30-sentence collection for a question with <strong>4 relevant</strong> sentences.</p>"
              "<ul><li><strong>Student One</strong> sets k = 8 and gets all 4: precision 50%, recall 100%.</li>"
              "<li><strong>Student Two</strong> sets k = 5 and gets 3: precision 60%, recall 75%.</li></ul>"
              "<p>Student Two says their search did the better job, since their precision is higher. Is that fair?</p>",
  [("No — Student One reached both ceilings (50% and 100%); Student Two missed both (80% and 100%)", None),
   ("Yes — higher precision means a better search", "Raw scores mean little without their ceilings. Student Two fell short of what k = 5 allowed."),
   ("Yes — half of Student One's results were junk", "That junk was forced by k = 8: only 4 relevant sentences exist. It's a reason to lower k, not a failed search."),
   ("No — Student One's precision is really 4 ÷ 4 = 100%", "4 ÷ 4 divides by relevant — that's recall. Precision divides by k: 4 ÷ 8.")],
  "<p>Student One: best hits = smaller of 8 and 4 = 4, so best precision = 4 ÷ 8 = 50% and best recall = 100%. They hit <strong>both</strong>.</p>"
  "<p>Student Two: best hits = smaller of 5 and 4 = 4, so best precision = 4 ÷ 5 = 80% and best recall = 100%. "
  "They got 60% and 75% — short on both.</p>")


# ------------------------------------------------------------------ assemble + check
LETTERS = "ABCD"
tests = {t: [] for t in range(1, 6)}
letter_count = {L: 0 for L in LETTERS}

for item in Q:
    ch = item["choices"]
    assert len(ch) == 4, item["prompt"]
    assert ch[0][1] is None and all(w for _, w in ch[1:]), item["prompt"]
    assert len({c for c, _ in ch}) == 4, "duplicate choice: " + item["prompt"]
    k = LETTERS.index(item["pos"])
    ordered = ch[1:k + 1] + [ch[0]] + ch[k + 1:]
    letter_count[item["pos"]] += 1
    out = dict(topic=item["topic"], prompt=item["prompt"],
               choices=[c for c, _ in ordered], answer=k,
               why=[w for _, w in ordered], solution=item["solution"])
    if item["doc"]:
        text, sep, size, overlap, expected = item["doc"]
        check_chunks(text, sep, size, overlap, expected)
        out["doc"] = dict(text=text, sep=sep, size=size, overlap=overlap)
        if item["settings"]:
            out["settings"] = item["settings"]
    tests[item["test"]].append(out)

for t, qs in tests.items():
    assert len(qs) == 10, f"test {t} has {len(qs)} questions"
    topics = [x["topic"] for x in qs]
    assert all(topics.count(tp) == 2 for tp in (VEC, DOT, CHK, EDGE, PR)), f"test {t} topic mix"

assert max(letter_count.values()) - min(letter_count.values()) <= 1, letter_count

# The body serif draws ‖ as a thin single bar; give it a font that shows two.
def norm_bars(x):
    if isinstance(x, str): return x.replace("‖", '<span class="nm">‖</span>')
    if isinstance(x, list): return [norm_bars(y) for y in x]
    if isinstance(x, dict): return {k: (y if k == "doc" else norm_bars(y)) for k, y in x.items()}
    return x

data = [dict(number=t, questions=norm_bars(qs)) for t, qs in tests.items()]
with open(OUT, "w", encoding="utf-8") as f:
    f.write("/* GENERATED by tools/build_questions.py — do not edit by hand. */\n")
    f.write("window.PRACTICE_TESTS = ")
    json.dump(data, f, ensure_ascii=False, indent=1)
    f.write(";\n")

print(f"OK: {len(Q)} questions, 5 tests, answer letters {letter_count}, "
      f"chunking checked against {'REAL langchain splitter' if HAVE_LANGCHAIN else 'built-in algorithm only'}")
print("wrote", os.path.normpath(OUT))
