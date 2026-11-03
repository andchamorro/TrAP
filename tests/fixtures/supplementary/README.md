# Fixtures for `scripts/make_supplementary_tables.py`

`results/` mirrors the layout the script reads. Every file is derived from the real
result tables with **every floating-point cell perturbed**, so the numbers here are
fabricated: some R² values exceed 1, and nothing in this directory may be quoted or
compared against the manuscript. Integer counts and labels are unchanged because the
script only formats them. The manifests keep only the `seed` and `params` it reads.

The point of the fixtures is to pin the *formatting*: column order, rounding,
scientific notation for small p values, caption numbering and the join between tables.

`expected_supplementary.md` is the golden output. After an intentional change to the
script's formatting, regenerate it with

    UPDATE_GOLDEN=1 pytest tests/scripts/test_make_supplementary_tables.py

and review the diff before committing.
