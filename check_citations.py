"""Citation consistency validator. Uses only the Python standard library."""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\s*\()")
_REF_LINE = re.compile(r"^\s*\[(\d+)\](?:\s+|$)")
_URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)


def _citation_numbers(group):
    """Expand comma-separated citation numbers and ranges without unbounded allocation."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if not span:
            numbers.append(int(part))
            continue
        start, end = map(int, span.groups())
        if end < start:
            numbers.extend((start, end))
        elif end - start <= 500:
            numbers.extend(range(start, end + 1))
        else:
            numbers.extend((start, end))
    return numbers


def _body_citations(body):
    """Find bracket citations outside inline/fenced code and Markdown links."""
    parts = _CODE.split(body)
    citations = set()
    for index, part in enumerate(parts):
        if index % 2:
            continue
        for match in _CITATION.finditer(part):
            citations.update(_citation_numbers(match.group(1)))
    return citations


def _reference_urls(line):
    urls = []
    for match in _URL.finditer(line):
        url = match.group(0).rstrip(".,;:!?\"")
        urls.append(url)
    return urls


def check(report_text, sources):
    """Return every citation/catalog/reference inconsistency found in the report."""
    problems = []
    if not isinstance(sources, list):
        return ["sources.json must contain a JSON list"]
    if not sources:
        return ["no sources in sources.json"]

    by_number = {}
    seen_urls = {}
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} must be an object")
            continue

        number = source.get("n")
        valid_number = isinstance(number, int) and not isinstance(number, bool)
        if not valid_number:
            problems.append(f"source entry {index} has non-integer n")
        elif number in by_number:
            problems.append(f"duplicate source number [{number}] in sources.json")
        else:
            by_number[number] = source

        url = source.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{number}] URL must start with http:// or https://")
        elif url in seen_urls:
            problems.append(f"duplicate URL in sources.json for [{number}] and [{seen_urls[url]}]")
        else:
            seen_urls[url] = number

    heading_matches = list(_REF_HEADING.finditer(str(report_text or "")))
    if not heading_matches:
        problems.append("report is missing the ## References heading")
        body = str(report_text or "")
        references = ""
    else:
        if len(heading_matches) > 1:
            problems.append("report must contain exactly one ## References heading")
        heading = heading_matches[0]
        body = str(report_text or "")[:heading.start()]
        references = str(report_text or "")[heading.end():]

    cited = _body_citations(body)
    known_numbers = set(by_number)
    for number in sorted(cited - known_numbers):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(known_numbers - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = []
    for line in references.splitlines():
        match = _REF_LINE.match(line)
        if match:
            reference_lines.append((int(match.group(1)), line))

    counts = {}
    for number, line in reference_lines:
        counts[number] = counts.get(number, 0) + 1
        if number not in by_number:
            problems.append(f"reference [{number}] has no matching source in sources.json")
        elif counts[number] > 1:
            problems.append(f"duplicate reference line for [{number}]")

        urls = _reference_urls(line)
        if len(urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL (found {len(urls)})")
        elif number in by_number:
            expected = by_number[number].get("url")
            if urls[0] != expected:
                problems.append(f"reference [{number}] URL does not match sources.json")

    for number in sorted(known_numbers):
        if counts.get(number, 0) == 0:
            problems.append(f"source [{number}] has no reference line")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
