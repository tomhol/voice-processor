#!/usr/bin/env python3
"""
optimize_transcription_for_tts.py

Reads a YAML transcription file (list of {text, start, end} segments)
and produces a compacted version suited for TTS engines that struggle
with very short text segments.

Merging rules (in priority order):
  1. Break at long gaps between segments (--min-gap-for-break, default 3s)
  2. Break at sentence-ending punctuation (. ? !)
  3. Break at pause punctuation (, ; :) when approaching the word limit
  4. Break when segment would exceed --max-words-in-segment
  5. If a single input segment already exceeds --max-words-in-segment, keep it as-is
"""

import argparse
import sys
import yaml


def count_words(text: str) -> int:
    return len(text.split())


def ends_with_sentence_end(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith(('.', '!', '?'))


def ends_with_pause(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith((',', ';', ':'))


def merge_segments(segments: list[dict], max_words: int, min_gap: float) -> list[dict]:
    """
    Merge input segments into larger chunks according to the spec.
    Returns a list of merged segment dicts with keys: start, text, end.

    Priority:
      1. Long gap  → always break (configurable threshold)
      2. Word limit exceeded → always break; oversized single input kept as-is
      3. Sentence-end punctuation (. ! ?) → preferred break, but only when the
         accumulator already has >= break_fraction of max_words words
      4. Pause punctuation (, ; :) → preferred break at the same threshold
      5. Otherwise: keep merging
    """
    if not segments:
        return []

    # Only prefer punctuation breaks once we've accumulated this fraction of the limit.
    # This prevents every short "Arms up." from becoming its own output segment.
    BREAK_FRACTION = 0.5

    merged = []

    acc_start: float = segments[0]['start']
    acc_text: str = segments[0]['text']
    acc_end: float = segments[0]['end']

    for i in range(1, len(segments)):
        seg = segments[i]
        gap = seg['start'] - acc_end
        combined_text = acc_text + ' ' + seg['text']
        combined_words = count_words(combined_text)
        acc_words = count_words(acc_text)

        # Rule 1: long gap → forced break
        if gap >= min_gap:
            merged.append({'start': acc_start, 'text': acc_text, 'end': acc_end})
            acc_start = seg['start']
            acc_text = seg['text']
            acc_end = seg['end']
            continue

        # Rule 2: merging would exceed word limit → flush and start fresh
        if combined_words > max_words:
            merged.append({'start': acc_start, 'text': acc_text, 'end': acc_end})
            acc_start = seg['start']
            acc_text = seg['text']
            acc_end = seg['end']
            continue

        # Rules 3 & 4: punctuation-based preferred break, only when substantial
        substantial = acc_words >= max(1, int(max_words * BREAK_FRACTION))
        if substantial:
            if ends_with_sentence_end(acc_text):
                merged.append({'start': acc_start, 'text': acc_text, 'end': acc_end})
                acc_start = seg['start']
                acc_text = seg['text']
                acc_end = seg['end']
                continue
            if ends_with_pause(acc_text):
                merged.append({'start': acc_start, 'text': acc_text, 'end': acc_end})
                acc_start = seg['start']
                acc_text = seg['text']
                acc_end = seg['end']
                continue

        # Otherwise: merge
        acc_text = combined_text
        acc_end = seg['end']

    merged.append({'start': acc_start, 'text': acc_text, 'end': acc_end})
    return merged


def format_yaml(segments: list[dict]) -> str:
    """
    Produce YAML output with attribute order: start, text, end.
    Uses a manual formatter to guarantee key order and clean multiline text.
    """
    lines = []
    for seg in segments:
        start = seg['start']
        text = seg['text']
        end = seg['end']

        # Format floats: strip trailing zeros but keep at least one decimal place
        def fmt(v: float) -> str:
            s = f'{v:.3f}'.rstrip('0')
            if s.endswith('.'):
                s += '0'
            return s

        lines.append(f'- start: {fmt(start)}')

        # For text: use block scalar if it contains a colon or is very long,
        # otherwise a plain quoted string.
        if ':' in text or len(text) > 80:
            # Use YAML literal block scalar
            indented = text.replace('\n', '\n    ')
            lines.append(f'  text: |-')
            lines.append(f'    {indented}')
        else:
            # Escape double-quotes inside the text
            escaped = text.replace('"', '\\"')
            lines.append(f'  text: "{escaped}"')

        lines.append(f'  end: {fmt(end)}')

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(
        description='Optimize a transcription YAML for TTS by merging short segments.'
    )
    parser.add_argument(
        'input',
        help='Path to the input YAML file (or "-" for stdin).'
    )
    parser.add_argument(
        '-o', '--output',
        default='-',
        help='Path to the output YAML file (default: stdout).'
    )
    parser.add_argument(
        '--max-words-in-segment',
        type=int,
        default=20,
        metavar='N',
        help='Maximum number of words allowed in a merged segment (default: 20).'
    )
    parser.add_argument(
        '--min-gap-for-break',
        type=float,
        default=3.0,
        metavar='SECONDS',
        help='Gap duration (seconds) that forces a segment break (default: 3.0).'
    )

    args = parser.parse_args()

    # --- Load input ---
    if args.input == '-':
        raw = sys.stdin.read()
    else:
        with open(args.input, 'r', encoding='utf-8') as fh:
            raw = fh.read()

    segments = yaml.safe_load(raw)

    if not isinstance(segments, list):
        sys.exit('ERROR: Input YAML must be a list of segment objects.')

    # Normalise: text may be a multi-line string (folded by PyYAML); collapse to single line.
    for seg in segments:
        seg['text'] = ' '.join(seg['text'].split())

    # --- Merge ---
    merged = merge_segments(segments, args.max_words_in_segment, args.min_gap_for_break)

    # --- Output ---
    output_yaml = format_yaml(merged)

    if args.output == '-':
        sys.stdout.write(output_yaml)
    else:
        with open(args.output, 'w', encoding='utf-8') as fh:
            fh.write(output_yaml)
        print(f'Written {len(merged)} segments to {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
