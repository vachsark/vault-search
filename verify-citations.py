#!/usr/bin/env python3
"""
verify-citations.py — Verify URLs/citations in markdown files.
Catches dead links, redirects, and hallucinated references.

Usage:
    python3 _scripts/verify-citations.py path/to/file.md
    python3 _scripts/verify-citations.py path/to/file.md --fix
    python3 _scripts/verify-citations.py --dir Knowledge/
    python3 _scripts/verify-citations.py --dir Knowledge/ --json

No external dependencies — stdlib only (urllib, concurrent.futures).
"""

import re
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class UrlResult:
    url: str
    status: str          # "ok" | "redirect" | "not_found" | "timeout" | "error" | "skipped" | "hallucinated"
    code: Optional[int] = None
    redirect_to: Optional[str] = None
    error_msg: Optional[str] = None
    hallucination_reason: Optional[str] = None
    source_file: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class FileReport:
    path: str
    total_urls: int = 0
    ok: list = field(default_factory=list)
    redirects: list = field(default_factory=list)
    not_found: list = field(default_factory=list)
    timeouts: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    hallucinated: list = field(default_factory=list)
    access_denied: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    @property
    def has_issues(self):
        return bool(self.not_found or self.timeouts or self.errors or self.hallucinated)

    @property
    def issue_count(self):
        return len(self.not_found) + len(self.timeouts) + len(self.errors) + len(self.hallucinated)


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

# Markdown link: [text](url)
RE_MD_LINK = re.compile(r'\[(?:[^\]]*)\]\(([^)]+)\)')
# Bare URLs
RE_BARE_URL = re.compile(r'(?<!\()\bhttps?://[^\s\)\]">]+')
# Wikilinks — must NOT be treated as URLs
RE_WIKILINK = re.compile(r'\[\[([^\]]+)\]\]')


def extract_urls(text: str) -> list[tuple[str, int]]:
    """Return list of (url, line_number) tuples. 1-indexed lines."""
    urls = []
    seen = set()
    lines = text.splitlines()

    for lineno, line in enumerate(lines, start=1):
        # Skip wikilink-only lines to avoid confusion, but still parse them
        # Collect markdown links
        for m in RE_MD_LINK.finditer(line):
            raw = m.group(1).strip()
            # Strip trailing punctuation that snuck in
            raw = raw.rstrip('.,;:)')
            if raw and raw not in seen:
                seen.add(raw)
                urls.append((raw, lineno))

        # Collect bare URLs (not already inside markdown link parens)
        for m in RE_BARE_URL.finditer(line):
            raw = m.group(0).rstrip('.,;:)')
            if raw and raw not in seen:
                seen.add(raw)
                urls.append((raw, lineno))

    return urls


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

# arxiv IDs should be YYMM.NNNNN (5 digits after dot, or 4 older ones)
# Matches /abs/, /html/, /pdf/, /src/ path variants
RE_ARXIV_ID = re.compile(r'arxiv\.org/(?:abs|html|pdf|src)/(\d{4}\.\d{4,5}(?:v\d+)?)', re.I)
RE_ARXIV_URL = re.compile(r'arxiv\.org', re.I)

# DOI patterns
RE_DOI_URL = re.compile(r'doi\.org/(.+)', re.I)
RE_DOI_PLACEHOLDER = re.compile(
    r'doi\.org/(10\.(XXXX|0000|1234)\b|.*placeholder.*|.*example.*)',
    re.I
)

# Placeholder patterns in any URL
RE_PLACEHOLDER = re.compile(
    r'example\.com|placeholder|your-?url|INSERT|FIXME|TODO|<[^>]+>',
    re.I
)

# Suspiciously sequential/round IDs (e.g. /1234567890, /0000000000)
RE_ROUND_ID = re.compile(r'/(0{6,}|1234567890|9876543210)(?:[/?#]|$)')

# Fake/malformed arxiv ID: wrong year (future), or not matching format
def _check_arxiv(url: str) -> Optional[str]:
    if not RE_ARXIV_URL.search(url):
        return None
    m = RE_ARXIV_ID.search(url)
    if not m:
        # arxiv URL but no valid ID
        return "arxiv URL missing valid YYMM.NNNNN ID"
    arxiv_id = m.group(1)
    yymm = arxiv_id[:4]
    try:
        year = int(yymm[:2])
        month = int(yymm[2:])
        # Current year is 26 (2026). Flag if year > 26 or month > 12
        if month < 1 or month > 12:
            return f"arxiv ID has invalid month: {arxiv_id}"
        if year > 26:  # update this ceiling as time passes
            return f"arxiv ID year {year} is in the future: {arxiv_id}"
    except ValueError:
        return f"arxiv ID malformed: {arxiv_id}"
    return None


def check_hallucination(url: str) -> Optional[str]:
    """Return a reason string if the URL looks hallucinated, else None."""
    if RE_PLACEHOLDER.search(url):
        return f"URL contains placeholder text"

    if RE_ROUND_ID.search(url):
        return f"URL contains suspiciously round/sequential ID"

    arxiv_issue = _check_arxiv(url)
    if arxiv_issue:
        return arxiv_issue

    if RE_DOI_PLACEHOLDER.search(url):
        return "DOI URL contains placeholder publisher code"

    return None


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

SKIP_SCHEMES = {'file', 'mailto', 'ftp'}
SKIP_HOSTS = {'localhost', '127.0.0.1', '0.0.0.0', '::1'}


def should_skip(url: str) -> Optional[str]:
    """Return skip reason string if URL should not be fetched."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return "unparseable URL"

    scheme = parsed.scheme.lower()
    if scheme in SKIP_SCHEMES:
        return f"skipped ({scheme}:// scheme)"

    if not scheme.startswith('http'):
        return f"skipped (non-http scheme: {scheme})"

    host = parsed.netloc.split(':')[0].lower()
    if host in SKIP_HOSTS:
        return f"skipped (local address: {host})"

    return None


# ---------------------------------------------------------------------------
# HTTP check
# ---------------------------------------------------------------------------

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (compatible; vault-citation-checker/1.0; '
        '+https://github.com/vachsark)'
    )
}

def check_url(url: str, timeout: int = 10) -> UrlResult:
    """Perform HEAD (then GET fallback) and return a UrlResult."""
    result = UrlResult(url=url, status="error")

    skip_reason = should_skip(url)
    if skip_reason:
        result.status = "skipped"
        result.error_msg = skip_reason
        return result

    halluc = check_hallucination(url)
    if halluc:
        result.status = "hallucinated"
        result.hallucination_reason = halluc
        return result

    req = urllib.request.Request(url, headers=HEADERS, method='HEAD')

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result.code = resp.status
            final_url = resp.geturl()
            if final_url != url and final_url.rstrip('/') != url.rstrip('/'):
                # Ignore cookie/auth-gate redirects (Nature, Springer bot detection)
                # Pattern: redirect to same base URL with error= or cookies_not_supported param
                parsed_orig = urllib.parse.urlparse(url)
                parsed_final = urllib.parse.urlparse(final_url)
                qs = urllib.parse.parse_qs(parsed_final.query)
                is_cookie_redirect = (
                    parsed_orig.netloc == parsed_final.netloc
                    and parsed_orig.path == parsed_final.path
                    and ('error' in qs or 'cookies_not_supported' in parsed_final.query)
                )
                if is_cookie_redirect:
                    result.status = "ok"
                    result.code = resp.status
                else:
                    result.status = "redirect"
                    result.redirect_to = final_url
            else:
                result.status = "ok"

    except urllib.error.HTTPError as e:
        if e.code == 405:
            # Server doesn't support HEAD — try GET with range
            req2 = urllib.request.Request(
                url,
                headers={**HEADERS, 'Range': 'bytes=0-0'},
                method='GET'
            )
            try:
                with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                    result.code = resp2.status
                    final_url = resp2.geturl()
                    if final_url != url and final_url.rstrip('/') != url.rstrip('/'):
                        result.status = "redirect"
                        result.redirect_to = final_url
                    else:
                        result.status = "ok"
            except urllib.error.HTTPError as e2:
                result.code = e2.code
                result.status = "not_found" if e2.code == 404 else "error"
                result.error_msg = str(e2)
            except Exception as e2:
                result.status = "error"
                result.error_msg = str(e2)
        elif e.code in (404, 410):
            result.code = e.code
            result.status = "not_found"
            result.error_msg = str(e)
        elif e.code == 403:
            # 403 often means paywall/bot-block, not a dead link
            result.code = e.code
            result.status = "access_denied"
            result.error_msg = "403 Forbidden (paywall or bot protection — likely real)"
        else:
            result.code = e.code
            result.status = "error"
            result.error_msg = str(e)

    except urllib.error.URLError as e:
        if 'timed out' in str(e).lower() or 'timeout' in str(e).lower():
            result.status = "timeout"
        else:
            result.status = "error"
        result.error_msg = str(e.reason)

    except TimeoutError:
        result.status = "timeout"
        result.error_msg = "connection timed out"

    except Exception as e:
        result.status = "error"
        result.error_msg = str(e)

    return result


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(path: Path, workers: int = 10, timeout: int = 10) -> FileReport:
    report = FileReport(path=str(path))

    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        report.errors.append(UrlResult(url="<file read error>", status="error", error_msg=str(e)))
        return report

    url_lines = extract_urls(text)
    report.total_urls = len(url_lines)

    if not url_lines:
        return report

    # Run checks concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for url, lineno in url_lines:
            fut = pool.submit(check_url, url, timeout)
            futures[fut] = (url, lineno)

        STATUS_TO_FIELD = {
            "ok": "ok",
            "redirect": "redirects",
            "not_found": "not_found",
            "timeout": "timeouts",
            "error": "errors",
            "hallucinated": "hallucinated",
            "access_denied": "access_denied",
            "skipped": "skipped",
        }

        for fut in as_completed(futures):
            url, lineno = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = UrlResult(url=url, status="error", error_msg=str(e))

            result.source_file = str(path)
            result.line_number = lineno

            field_name = STATUS_TO_FIELD.get(result.status, "errors")
            getattr(report, field_name).append(result)

    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

STATUS_ICON = {
    'ok': 'OK',
    'redirect': 'REDIR',
    'not_found': 'DEAD',
    'timeout': 'TIMEOUT',
    'error': 'ERROR',
    'hallucinated': 'HALLUC',
    'skipped': 'SKIP',
}

def print_report(report: FileReport, verbose: bool = False, show_ok: bool = False):
    total = report.total_urls
    issues = report.issue_count
    health = "CLEAN" if not report.has_issues else "ISSUES FOUND"

    print(f"\n{'='*60}")
    print(f"  {report.path}")
    print(f"  {total} URLs checked | {health}")
    print(f"{'='*60}")

    def _print_group(label: str, items: list, detail_fn=None):
        if not items:
            return
        print(f"\n[{label}] ({len(items)})")
        for r in items:
            loc = f"L{r.line_number}" if r.line_number else ""
            url_display = r.url[:80] + ('...' if len(r.url) > 80 else '')
            print(f"  {loc:>5}  {url_display}")
            if detail_fn:
                detail = detail_fn(r)
                if detail:
                    print(f"         {detail}")

    _print_group(
        "HALLUCINATED",
        report.hallucinated,
        lambda r: r.hallucination_reason
    )
    _print_group(
        "DEAD (404/410)",
        report.not_found,
        lambda r: f"HTTP {r.code}"
    )
    _print_group(
        "TIMEOUT",
        report.timeouts,
        lambda r: r.error_msg
    )
    _print_group(
        "ERROR",
        report.errors,
        lambda r: r.error_msg
    )
    _print_group(
        "REDIRECT",
        report.redirects,
        lambda r: f"-> {r.redirect_to}"
    )

    if show_ok:
        _print_group("OK", report.ok, lambda r: f"HTTP {r.code}")

    if verbose:
        _print_group(
            "ACCESS DENIED (403 — likely real, bot-blocked)",
            report.access_denied,
            lambda r: r.error_msg
        )
        _print_group("SKIPPED", report.skipped, lambda r: r.error_msg)

    if not report.has_issues and not report.redirects:
        if report.access_denied and not verbose:
            print(f"\n  URLs verified OK. ({len(report.access_denied)} bot-blocked, use --verbose to see)")
        else:
            print("\n  All URLs verified OK.")


def print_summary(reports: list[FileReport]):
    total_files = len(reports)
    total_urls = sum(r.total_urls for r in reports)
    total_issues = sum(r.issue_count for r in reports)
    files_with_issues = sum(1 for r in reports if r.has_issues)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Files scanned : {total_files}")
    print(f"  Total URLs    : {total_urls}")
    print(f"  Files w/issues: {files_with_issues}")
    print(f"  Total issues  : {total_issues}")

    all_hallucinated = sum(len(r.hallucinated) for r in reports)
    all_dead = sum(len(r.not_found) for r in reports)
    all_timeouts = sum(len(r.timeouts) for r in reports)
    all_errors = sum(len(r.errors) for r in reports)
    all_redirects = sum(len(r.redirects) for r in reports)
    all_access_denied = sum(len(r.access_denied) for r in reports)

    if total_issues:
        if all_hallucinated: print(f"  Hallucinated  : {all_hallucinated}")
        if all_dead:         print(f"  Dead (404)    : {all_dead}")
        if all_timeouts:     print(f"  Timeouts      : {all_timeouts}")
        if all_errors:       print(f"  Errors        : {all_errors}")
        if all_redirects:    print(f"  Redirects     : {all_redirects}")
    if all_access_denied:
        print(f"  Bot-blocked   : {all_access_denied} (403, likely real — use --verbose)")


def suggest_fixes(reports: list[FileReport]):
    """Print --fix suggestions (redirect targets as replacements)."""
    print(f"\n{'='*60}")
    print(f"  SUGGESTED FIXES")
    print(f"{'='*60}")
    any_fix = False
    for report in reports:
        for r in report.redirects:
            if r.redirect_to:
                print(f"\n  File: {report.path}  L{r.line_number}")
                print(f"  OLD: {r.url}")
                print(f"  NEW: {r.redirect_to}")
                any_fix = True
        for r in report.hallucinated:
            print(f"\n  File: {report.path}  L{r.line_number}")
            print(f"  URL: {r.url}")
            print(f"  ISSUE: {r.hallucination_reason}")
            print(f"  FIX: Remove or replace with a verified URL")
            any_fix = True
    if not any_fix:
        print("  No automated fixes available.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_markdown_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.md"))


def main():
    parser = argparse.ArgumentParser(
        description="Verify citations/URLs in markdown files. Detects dead links and hallucinated references."
    )
    parser.add_argument(
        "file",
        nargs="*",
        type=Path,
        help="One or more markdown files to check"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        metavar="DIR",
        help="Scan all .md files in a directory (recursive)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Print suggested replacements for redirects and hallucinated URLs"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also show skipped URLs"
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Include working URLs in report"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Concurrent workers (default: 10)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Per-URL timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--issues-only",
        action="store_true",
        help="Only print files that have issues (suppress clean files)"
    )

    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.print_help()
        sys.exit(1)

    # Collect files
    files: list[Path] = []
    for f in (args.file or []):
        if not f.exists():
            print(f"Error: {f} not found", file=sys.stderr)
            sys.exit(1)
        files.append(f)
    if args.dir:
        if not args.dir.is_dir():
            print(f"Error: {args.dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        files.extend(collect_markdown_files(args.dir))

    if not files:
        print("No markdown files found.")
        sys.exit(0)

    # Process
    reports: list[FileReport] = []
    for i, f in enumerate(files, 1):
        if len(files) > 1:
            print(f"Checking ({i}/{len(files)}): {f}", file=sys.stderr)
        report = process_file(f, workers=args.workers, timeout=args.timeout)
        reports.append(report)

    # Output
    if args.json:
        output = []
        for r in reports:
            d = asdict(r)
            output.append(d)
        print(json.dumps(output, indent=2))
        return

    for report in reports:
        if args.issues_only and not report.has_issues:
            continue
        print_report(report, verbose=args.verbose, show_ok=args.show_ok)

    if len(reports) > 1:
        print_summary(reports)

    if args.fix:
        suggest_fixes(reports)

    # Exit code: 1 if any issues found
    if any(r.has_issues for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
