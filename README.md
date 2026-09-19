# Math for AI — Practice Tests

Five 10-question multiple-choice practice tests for CIS-170 (Mathematics for Data Science,
Mercyhurst, Fall 2026). Static site, no build step for the pages — GitHub Pages serves it as is.

- `index.html` — home page: the five tests + links to the solutions
- `test.html?t=N` — take test N. Submitting shows the score and marks missed questions
  **without revealing the correct answer**. Students can retry just the missed ones.
- `solutions.html?t=N` — worked solutions, including why each wrong choice is wrong
- `questions.js` — **generated**; do not edit by hand
- `tools/build_questions.py` — the source of truth for all 50 questions

## Editing questions

Edit `tools/build_questions.py`, then rebuild:

    ~/.langflow/.langflow-venv/bin/python tools/build_questions.py   # also checks chunking against the real Langflow splitter
    python3 tools/build_questions.py                                 # works anywhere; built-in splitter check only

The build asserts every answer (vector math, cosine, precision/recall, and every chunk length)
before writing `questions.js`, so a typo in an answer key fails loudly.

## Notes

- The answer key is in `questions.js`, so a determined student can find it in the page source.
  That's acceptable here — the solutions are published anyway. The point is that the results
  screen never shows them.
- Progress and "best score" live in the student's browser (localStorage). Nothing is sent anywhere.
- All numbers are chosen to be done with pencil and paper — no calculator, same as the real exams.
