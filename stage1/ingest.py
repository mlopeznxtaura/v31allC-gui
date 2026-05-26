"""stage1/ingest.py -- Parse various data formats, with permanent JSON repair."""

import json
import csv
import re
from io import StringIO


class Stage1Ingest:
    """Ingest raw data in various formats"""

    @staticmethod
    def _repair_json(raw: str) -> str:
        """
        Permanent best-effort repair for malformed JSON arrays/objects.
        Fixes the most common real-world corruption:
          1. Missing comma between adjacent top-level objects:  }\n{  -> },\n{
          2. Windows CRLF line endings
          3. UTF-8 BOM
          4. Stray ASCII control characters
        Safe to call on already-valid JSON (no-op in that case).
        """
        raw = raw.lstrip('\ufeff')                          # BOM
        raw = raw.replace('\r\n', '\n').replace('\r', '\n') # CRLF
        raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)  # ctrl chars
        # Insert missing comma between  }[whitespace]{
        raw = re.sub(r'(\})(\s*\n\s*)(\{)', r'\1,\2\3', raw)
        raw = re.sub(r'(\])(\s*\n\s*)(\[)', r'\1,\2\3', raw)
        return raw

    @staticmethod
    def detect_format(content: str) -> str:
        """Guess file format from content"""
        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])

        content = content.strip()
        first_line = content.split('\n')[0].strip()

        if first_line.startswith('{'):
            json_lines = sum(1 for line in content.split('\n') if line.strip().startswith('{'))
            total_lines = len([l for l in content.split('\n') if l.strip()])
            if json_lines > 1 and total_lines > 1:
                return 'jsonl'
            else:
                return 'json'

        if content.startswith('['):
            return 'json'

        if '\t' in content or ',' in content:
            return 'csv'

        return 'text'

    @staticmethod
    def parse_jsonl(content: str) -> list:
        """Parse JSONL (one JSON object per line)"""
        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])

        records = []
        for line in content.strip().split('\n'):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        return records

    @staticmethod
    def parse_json(content: str) -> list:
        """
        Parse JSON array or single object.
        Attempt order:
          1. Direct parse (fast path for valid JSON)
          2. _repair_json then parse  (handles missing commas, BOM, CRLF)
          3. Regex object extraction  (last resort for deeply broken files)
        """
        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])

        # 1. Direct
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass

        # 2. Repair
        repaired = Stage1Ingest._repair_json(content)
        try:
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass

        # 3. Regex extraction of individual objects (deeply broken files)
        records = []
        depth = 0
        start = None
        for i, ch in enumerate(repaired):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    fragment = repaired[start:i+1]
                    try:
                        records.append(json.loads(fragment))
                    except Exception:
                        pass
                    start = None
        return records

    @staticmethod
    def parse_csv(content: str) -> list:
        """Parse CSV into list of dicts"""
        reader = csv.DictReader(StringIO(content))
        return list(reader)

    @staticmethod
    def parse_text(content: str) -> list:
        """Parse plain text as single record"""
        return [{"text": content}]

    @staticmethod
    def parse_format(content: str, fmt: str) -> list:
        """Route to appropriate parser"""
        if fmt == 'jsonl':
            return Stage1Ingest.parse_jsonl(content)
        elif fmt == 'json':
            return Stage1Ingest.parse_json(content)
        elif fmt == 'csv':
            return Stage1Ingest.parse_csv(content)
        else:
            return Stage1Ingest.parse_text(content)