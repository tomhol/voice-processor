#!/usr/bin/env python3
"""
normalize_transcription.py

Re-segments a YAML transcription file so that each output segment corresponds
to a natural sentence (detected by a period '.').  When combining multiple
input segments, the start of the first and end of the last are used.  When
splitting a single segment, timestamps are linearly interpolated by character
count, and a configurable gap is inserted between the new segments.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import List

import yaml


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    text: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """Split *text* on sentence-ending periods.

    Keeps the period attached to the sentence that ends with it.  Trailing
    whitespace / leading whitespace around each sentence is stripped.
    Empty strings are dropped.
    """
    # Split on '.' but keep the delimiter attached to the left part.
    parts = re.split(r'(?<=\.)\s*', text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _char_offset(text: str, char_index: int) -> float:
    """Return the fraction [0, 1] of *char_index* within *text*."""
    length = len(text)
    if length == 0:
        return 0.0
    return min(char_index / length, 1.0)


def _split_segment(seg: Segment, max_words: int, gap: float) -> List[Segment]:
    """Split *seg* first by sentence boundaries, then by *max_words*.

    Timestamps for sub-segments are linearly interpolated by character
    position within the original text.  A *gap* (seconds) is inserted
    between the newly created segments by trimming it from the end of each
    sub-segment (except the last).
    """
    full_text = seg.text
    duration = seg.end - seg.start

    # Step 1: split into sentence-level chunks
    sentences = _split_into_sentences(full_text)
    if not sentences:
        return [seg]

    # Step 2: further split any sentence that exceeds max_words
    chunks: List[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
        else:
            # Slice the sentence into word-count-limited pieces
            for i in range(0, len(words), max_words):
                chunks.append(' '.join(words[i:i + max_words]))

    if len(chunks) == 1:
        # Nothing changed — return the original segment (possibly trimmed)
        return [Segment(text=chunks[0], start=seg.start, end=seg.end)]

    # Step 3: assign timestamps using character offset within full_text
    # Build a cumulative character position for the start of each chunk
    # inside *full_text* so we can interpolate times.
    results: List[Segment] = []
    search_pos = 0  # current scan position in full_text

    for idx, chunk in enumerate(chunks):
        # Locate the chunk inside full_text starting from search_pos.
        # We look for the first word of the chunk to anchor it.
        first_word = chunk.split()[0] if chunk.split() else chunk
        found = full_text.find(first_word, search_pos)
        if found == -1:
            found = search_pos  # fallback

        chunk_start_char = found
        chunk_end_char = chunk_start_char + len(chunk)
        search_pos = chunk_end_char  # advance past this chunk

        t_start = seg.start + _char_offset(full_text, chunk_start_char) * duration
        t_end = seg.start + _char_offset(full_text, chunk_end_char) * duration

        # Clamp to segment boundaries
        t_start = max(seg.start, min(t_start, seg.end))
        t_end = max(t_start, min(t_end, seg.end))

        # Apply gap: shorten end of non-last segments
        if idx < len(chunks) - 1 and gap > 0:
            t_end = max(t_start, t_end - gap)

        results.append(Segment(text=chunk, start=round(t_start, 3), end=round(t_end, 3)))

    return results


# ---------------------------------------------------------------------------
# Main normalisation logic
# ---------------------------------------------------------------------------

def normalize(segments: List[Segment], max_words: int, gap: float) -> List[Segment]:
    """Return a new list of segments aligned to sentence boundaries."""

    # Phase 1: merge all input segments into a single stream of (text, seg_ref)
    # pairs so we can detect sentence boundaries across segment borders.

    # We concatenate all texts (with a space between them) while tracking
    # which source segment owns each character.
    combined_text = ""
    # List of (char_start, char_end, segment) tuples
    char_map: List[tuple] = []  # (char_start, char_end, Segment)

    for seg in segments:
        start_char = len(combined_text)
        if combined_text and not combined_text.endswith(' '):
            combined_text += ' '
            start_char = len(combined_text)
        combined_text += seg.text
        end_char = len(combined_text)
        char_map.append((start_char, end_char, seg))

    def time_at_char(char_idx: int) -> float:
        """Interpolate a timestamp for *char_idx* in the combined text."""
        for (cs, ce, seg) in char_map:
            if cs <= char_idx <= ce:
                frac = (char_idx - cs) / max(ce - cs, 1)
                return seg.start + frac * (seg.end - seg.start)
        # Beyond the last segment
        return char_map[-1][2].end

    # Phase 2: split the combined text into sentences (by '.')
    sentences = _split_into_sentences(combined_text)

    # Phase 3: further split overlong sentences by max_words, then assign times
    output: List[Segment] = []
    search_pos = 0

    for sentence in sentences:
        words = sentence.split()

        # Locate the sentence in combined_text
        first_word = words[0] if words else sentence
        found = combined_text.find(first_word, search_pos)
        if found == -1:
            found = search_pos

        sentence_char_start = found
        sentence_char_end = sentence_char_start + len(sentence)
        search_pos = sentence_char_end

        if len(words) <= max_words:
            # Single output segment
            t_start = time_at_char(sentence_char_start)
            t_end = time_at_char(sentence_char_end)
            output.append(Segment(
                text=sentence,
                start=round(t_start, 3),
                end=round(t_end, 3),
            ))
        else:
            # Split into max_words-sized chunks
            sub_search = sentence_char_start
            for i in range(0, len(words), max_words):
                chunk_words = words[i:i + max_words]
                chunk = ' '.join(chunk_words)
                first_cw = chunk_words[0]
                chunk_found = combined_text.find(first_cw, sub_search)
                if chunk_found == -1:
                    chunk_found = sub_search

                chunk_char_start = chunk_found
                chunk_char_end = chunk_char_start + len(chunk)
                sub_search = chunk_char_end

                t_start = time_at_char(chunk_char_start)
                t_end = time_at_char(chunk_char_end)
                is_last_chunk = (i + max_words >= len(words))
                if not is_last_chunk and gap > 0:
                    t_end = max(t_start, t_end - gap)

                output.append(Segment(
                    text=chunk,
                    start=round(t_start, 3),
                    end=round(t_end, 3),
                ))

    return output


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> List[Segment]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    segments = []
    for item in data:
        segments.append(Segment(
            text=str(item['text']).strip(),
            start=float(item['start']),
            end=float(item['end']),
        ))
    return segments


def dump_yaml(segments: List[Segment]) -> str:
    data = [{'text': s.text, 'start': s.start, 'end': s.end} for s in segments]
    return yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a YAML transcription file so that each segment "
            "corresponds to a complete sentence."
        )
    )
    parser.add_argument('input', help='Path to the input YAML file.')
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Path to the output YAML file.  Defaults to stdout.',
    )
    parser.add_argument(
        '--max-words-in-segment',
        type=int,
        default=30,
        dest='max_words',
        help='Maximum number of words per output segment when no sentence '
             'boundary is found (default: 30).',
    )
    parser.add_argument(
        '--gap-for-splitting',
        type=float,
        default=1.0,
        dest='gap',
        help='Gap in seconds to insert between segments created by splitting '
             'a single overlong segment (default: 1.0).',
    )

    args = parser.parse_args()

    segments = load_yaml(args.input)
    normalized = normalize(segments, max_words=args.max_words, gap=args.gap)
    result = dump_yaml(normalized)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Wrote {len(normalized)} segment(s) to '{args.output}'.")
    else:
        sys.stdout.write(result)


if __name__ == '__main__':
    main()
