Describes your simple baseline, gives sample output, and reports the score of the baseline when you run it on the test set, and evaluate it with your scoring script.

# Simple Baseline: Lexical Tool Matching + Rule-Based Argument Extraction

simple-baseline.py is a deliberately simple, non-neural baseline for the function-calling task (xLAM-style data: a natural-language query, a list of candidate tools, and gold answers consisting of one or more calls). It uses only the Python standard library.

## How it works

For each request the baseline:

1. Splits the request into clauses (roughly one clause per intended call) at sentence boundaries, at words like "also", "then", "plus", "as well as", and at "and" followed by a verb (e.g. ", and check ...").
2. Scores every candidate tool against each clause by IDF-weighted word overlap between the clause and the tool's name, description, and parameter names. Words are lower-cased, split on snake_case/camelCase, stripped of stopwords, and lightly stemmed.
3. Keeps the best tool per clause. A weak clause (score below min_score) is treated as a continuation of the previous clause, as is a clause that yields no argument values for a tool that needs some. At least one call is always produced.
4. Fills arguments with fixed rules. Values are read directly off the request text: lists, quoted strings, URLs, e-mails, IPv4 addresses, ISO dates (YYYY-MM-DD), number sequences, single numbers, and capitalized names. They are assigned to parameters by type compatibility and by nearby parameter-name cues (e.g. "radius 5"). Remaining required parameters take the next compatible value in schema order. Values are cast to the parameter's declared type.
5. Fans out one call per value when a tool has a single required parameter and the clause contains several values (e.g. "search for 'a' and 'b'" produces two calls).

## What is learned from training data

Only the inverse document frequency (IDF) of words over the distinct tools in the training set. This down-weights generic words such as "fetch" or "retrieve". Nothing else is trained.

## Tunable settings
| Setting | Meaning | Default |
| -------- | -------- | -------- |
| min_score    | Minimum tool score for a clause after the first  | 0.0     |
| segment      | Split request into clauses     | on     |
| fanout       | One call per repeated value for single-parameter tools  | on    |
|	fill_optional |  Also fill optional parameters |  on    |
	

Settings are chosen by grid search on the development split (--tune). The script drops answers and the metadata sidecar before predicting, so gold labels cannot leak into the rules.
