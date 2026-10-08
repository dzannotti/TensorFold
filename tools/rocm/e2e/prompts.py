"""The fixed prompt set the agreement reference and the contracts use: prose, code, math, structured, multilingual, tools."""
import random

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                     "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search_files", "description": "Search a repository for files matching a glob and containing a string.",
    "parameters": {"type": "object", "properties": {"glob": {"type": "string"}, "contains": {"type": "string"},
                                                     "max_results": {"type": "integer"}},
                   "required": ["glob", "contains"]}}}


def user(text):
    return [{"role": "user", "content": text}]


# name -> (messages, tools)
REF = {
    "prose-sky": (user("Explain in four sentences why the sky looks blue during the day and red at sunset."), None),
    "prose-story": (user("Write a short story opening (two paragraphs) about a lighthouse keeper who finds a letter."), None),
    "prose-hashmap": (user("Write a detailed step-by-step explanation of how a hash map works, including collision "
                           "handling, resizing, and time complexity. Be thorough."), None),
    "prose-email": (user("Draft a polite email asking a colleague to review a pull request by Friday."), None),
    "code-fib": (user("Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."), None),
    "code-quicksort": (user("Write a Python quicksort with docstring and tests."), None),
    "code-zig": (user("Write a Zig function that reverses a slice of u8 in place, with a test block."), None),
    "code-sql": (user("Write a SQL query that returns the top 5 customers by total order value in 2025, with "
                      "tables customers(id, name) and orders(id, customer_id, total, created_at)."), None),
    "code-bugfix": (user("Find and fix the bug:\n\ndef mean(xs):\n    total = 0\n    for x in xs:\n        total += x\n"
                         "    return total / len(xs) if xs else total / 0\n"), None),
    "code-cuda": (user("Write a CUDA kernel that adds two float vectors, plus the host code that launches it."), None),
    "math-primes": (user("List the first twelve prime numbers, then give their sum and show the addition."), None),
    "math-train": (user("A train leaves at 09:40 and arrives at 13:15. How long is the trip? Show the steps."), None),
    "math-algebra": (user("Solve for x: 3(x - 4) + 2 = 5x - 18. Show each step."), None),
    "json-book": (user("Write a short JSON object describing a library book with title, author, year and three "
                       "subject tags."), None),
    "table-countries": (user("Give a markdown table of 6 countries with capital, currency and continent."), None),
    "count": (user("Count from 1 to 120. Output only the numbers, separated by spaces. No other text."), None),
    "fr-translate": (user("Translate into French: 'The old bridge was repaired before the winter storms came to "
                          "the valley.' Then explain two word choices."), None),
    "zh-explain": (user("用三句话解释什么是机器学习。"), None),
    "tool-weather": (user("What's the weather like in Rome right now? Use celsius."), [WEATHER]),
    "tool-search": (user("Find the Python files under src/ that mention 'retry_backoff', at most 10."), [SEARCH]),
}

WORDS = ("river engine winter market signal garden bridge letter theory method paper field window voice harbor lantern "
         "orchard ledger quarry canyon meadow compass anchor beacon cellar chimney harvest island journey kettle "
         "ladder mirror needle pebble ribbon saddle timber tunnel valley wagon atlas barrel candle desert ember").split()


def filler(words: int, seed: int) -> str:
    """Seeded prose, ~1.3 tokens a word."""
    rng = random.Random(seed)
    out = []
    while len(out) < words:
        s = [rng.choice(WORDS) for _ in range(rng.randint(6, 14))]
        out += [s[0].capitalize()] + s[1:-1] + [s[-1] + "."]
    return " ".join(out)
