"""Observable features and answer candidates from the two frozen prompts."""

from dataclasses import dataclass
import math
import re
import unicodedata

from collect import SCIENCE_SUFFIX, canonical_answer


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
RATIONAL = re.compile(r'-?\d+(?:/[1-9]\d*)?')
OPTION = re.compile(r'^([A-J])\. ', re.MULTILINE)
CONCLUSION = re.compile(
    r'(?i:\b(?:final(?: answer)?\s*\**\s*:|(?:correct |best )?(?:answer|option|choice)\s*(?:is|:)'
    r'|(?:corresponds|matches)\s+to\s+(?:option|choice)))[\s*`([:]*([A-J])(?=\W|$)'
)
NEGATION = re.compile(r"\b(?:not|never|exclude\w*|incorrect|reject\w*|rather than)\b", re.IGNORECASE)
UNCERTAINTY = re.compile(r'\b(?:likely|seems|approximately|closest|similar|uncertain|maybe|assuming)\b', re.IGNORECASE)
MISMATCH = re.compile(r"\b(?:does not match|none of|not provided|not listed|typo|no solution|cannot determine)\b", re.IGNORECASE)
EQUATION = re.compile(rf'(?<![\d.])([\d.+*/() \-]{{3,}})\s*=\s*({NUMBER})(?![\d.])')
SCIENTIFIC = rf'{NUMBER}(?:\s*(?:[eE]\s*|(?:×|\\times|·)\s*10\s*(?:\^\s*)?\{{?)[+-]?\d+\}}?)?'
QUANTITY = re.compile(rf'^\s*\$?({SCIENTIFIC})\$?(?:\s+[a-zA-Zμ°%][\wμ°%/^. -]{{0,22}})?\s*\.?$')
RESULT = re.compile(rf'(?:≈|=|\b(?:approximately|yields?|gives?|equals?))\s*\$?({SCIENTIFIC})', re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    answer: str = ''
    source: str = ''


def choices(question):
    text = question.removesuffix(SCIENCE_SUFFIX)
    matches = list(OPTION.finditer(text))
    return {m[1]: text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
            for i, m in enumerate(matches)}


def words(text):
    return re.findall(r'\b\w+\b', unicodedata.normalize('NFKC', text).casefold())


def question_features(domain, subject, question):
    text = question.removesuffix(SCIENCE_SUFFIX)
    options = choices(question) if domain == 'science' else {}
    stem = text.split('\n\nOptions:\n', 1)[0]
    numeric = sum(bool(re.search(r'\d', option)) for option in options.values())
    return [
        domain == 'math', subject == 'biology', subject == 'chemistry', subject == 'physics',
        math.log1p(len(stem)), math.log1p(len(text)), math.log1p(len(words(stem))),
        len(re.findall(NUMBER, stem)), len(options), numeric,
        '%' in stem, bool(re.search(r'\b(?:except|not|least|false)\b', stem, re.IGNORECASE)),
        bool(re.search(r'\b(?:if|assuming|suppose)\b', stem, re.IGNORECASE)),
        bool(re.search(r'\b(?:ratio|percentage|percent|probability|average)\b', stem, re.IGNORECASE)),
        bool(re.search(r'\b(?:per|each|every)\b', stem, re.IGNORECASE)),
        '\n' in stem, '\\' in stem, bool(re.search(r'\d\.\d', stem)),
    ]


def math_checks(text):
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
    for before, after in (('×', '*'), ('÷', '/'), ('−', '-'), (r'\times', '*'), (r'\cdot', '*'),
                          (r'\div', '/'), ('$', ''), (r'\(', ''), (r'\)', ''), (',', '')):
        text = text.replace(before, after)
    checks = []
    for left, right in EQUATION.findall(text):
        expression = left.strip()
        if not re.search(r'\d\s*[-+*/]', expression) or len(expression) > 128:
            continue
        value, given = canonical_answer(expression), canonical_answer(right)
        if RATIONAL.fullmatch(value) and RATIONAL.fullmatch(given):
            checks.append((value, given))
    return checks


def science_support(text, options):
    """Lower scores are absent or incidental mentions. Higher scores favor conclusions."""
    support = {key: 0.0 for key in options}
    explicit = []
    for match in CONCLUSION.finditer(text):
        key = match[1]
        if key in support and not NEGATION.search(text[max(0, match.start() - 22):match.start()]):
            position = match.end() / max(len(text), 1)
            support[key] = max(support[key], 3.0 + position)
            explicit.append(key)
    stripped = text.strip('* `\n')
    standalone = re.match(r'^([A-J])(?:[.)]\s|$)', stripped)
    if standalone and standalone[1] in options and '\n' not in stripped:
        support[standalone[1]] = max(support[standalone[1]], 3.5)
        explicit.append(standalone[1])

    normalized = ' ' + ' '.join(words(text)) + ' '
    phrases = {key: ' '.join(words(value)) for key, value in options.items()}
    for key, phrase in phrases.items():
        value = options[key]
        if not phrase or list(phrases.values()).count(phrase) > 1:
            continue
        needle = ' ' + phrase + ' '
        location = normalized.rfind(needle)
        if location < 0 or NEGATION.search(normalized[max(0, location - 40):location]):
            continue
        strength = 1.0 if len(phrase) >= 10 else 0.25
        if stripped.casefold().rstrip('.') == value.strip().casefold().rstrip('.'):
            strength = 3.5
        elif re.search(r'\b(?:thus|therefore|yields?|gives?|final|answer)\b',
                       normalized[max(0, location - 48):location]):
            strength += 0.7
        support[key] = max(support[key], strength * (0.8 + 0.2 * location / max(len(normalized), 1)))
    return support, explicit


def numeric_science(text, options):
    """Map a reported quantitative result to a uniquely close numeric option."""
    def value(raw):
        normalized = re.sub(r'(?:×|\\times|·)\s*10\s*\^?\s*', 'e', raw)
        try:
            number = float(re.sub(r'[\s{}]', '', normalized))
            return number if math.isfinite(number) else None
        except ValueError:
            return None

    found = RESULT.findall(text[-300:])
    if not found:
        return ''
    observed = value(found[-1])
    if observed is None:
        return ''
    distances = []
    for label, option in options.items():
        matched = QUANTITY.fullmatch(option)
        number = value(matched[1]) if matched else None
        if number is not None:
            distances.append((abs(observed - number) / max(abs(observed), abs(number), 1e-200), label))
    distances.sort()
    if len(distances) >= 3 and distances[0][0] <= 0.04 and distances[1][0] - distances[0][0] >= 0.05:
        return distances[0][1]
    return ''


def recovery(domain, question, record):
    """Candidates must be present in the trace or derivable from explicit arithmetic."""
    text, answer = record['text'], record['answer']
    if domain == 'science':
        options = choices(question)
        support, explicit = science_support(text, options)
        ranked = sorted(support, key=lambda key: (-support[key], key))
        if explicit and explicit[-1] != answer:
            return Candidate(explicit[-1], 'explicit'), support
        numeric = numeric_science(text, options)
        if numeric and numeric != answer and not MISMATCH.search(text):
            return Candidate(numeric, 'arithmetic'), support
        if ranked and not answer and not MISMATCH.search(text):
            best = ranked[0]
            second = support[ranked[1]] if len(ranked) > 1 else 0.0
            if support[best] >= 1.0 and support[best] - second >= 0.3:
                return Candidate(best, 'phrase'), support
        return Candidate(), support

    checks = math_checks(text)
    for line in reversed(text.splitlines()):
        candidate = re.search(r'(?i)^\s*[*`]*(?:final answer|final|answer)\s*[:=]\s*([\d.()+*/$,% \-]+)', line)
        if not candidate:
            continue
        read = canonical_answer(candidate[1])
        if RATIONAL.fullmatch(read) and read != answer:
            return Candidate(read, 'explicit'), {}
        break
    if checks and not MISMATCH.search(text):
        expected, reported = checks[-1]
        if answer and answer == reported and expected != reported:
            return Candidate(expected, 'arithmetic'), {}
        if not answer and expected == reported and not NEGATION.search(text[-90:]):
            return Candidate(expected, 'arithmetic'), {}
    return Candidate(), {}


def compact_features(domain, question, record, candidate, support):
    text, answer = record['text'], record['answer']
    checks = math_checks(text)
    output = record['output_tokens']
    distinct = sorted(support.values(), reverse=True)
    first, second = (distinct + [0.0, 0.0])[:2]
    vocab = set(words(text))
    source = set(words(question.removesuffix(SCIENCE_SUFFIX)))
    question_numbers = set(re.findall(NUMBER, question.split('\n\nOptions:\n', 1)[0]))
    response_numbers = set(re.findall(NUMBER, text))
    return [
        bool(answer), record['finish'] == 'length', output / 128,
        output >= 48, output >= 80, output >= 110, record['input_tokens'] / 256,
        len(checks), sum(a == b for a, b in checks), sum(a != b for a, b in checks),
        bool(checks and checks[-1][1] == answer), bool(checks and checks[-1][0] == answer),
        bool(MISMATCH.search(text)), len(UNCERTAINTY.findall(text)), len(NEGATION.findall(text)),
        'final' in text.casefold(), len(vocab & source) / max(len(vocab), 1),
        len(text.splitlines()), len(re.findall(NUMBER, text)),
        bool(candidate.answer), candidate.source == 'explicit', candidate.source == 'arithmetic',
        first, first - second, support.get(answer, 0.0), sum(value > 1 for value in support.values()),
        domain == 'math' and '/' in answer, len(question_numbers & response_numbers) / max(len(question_numbers), 1),
        bool(re.fullmatch(r'-\d+', answer)), bool(re.search(r'\b(?:check|fix|mistake|rather than)\b', text, re.IGNORECASE)),
    ]


def short_features(domain, record):
    evidence = record['evidence']
    ranked = sorted(evidence.values(), reverse=True) if domain == 'science' else []
    best, second = (ranked + [0.0, 0.0])[:2]
    logprob = evidence.get('mean_logprob')
    return [bool(record['answer']), record['output_tokens'] / 32, record['input_tokens'] / 256,
            best, best - second, sum(value * value for value in ranked),
            math.exp(logprob) if logprob is not None else 0.0]
