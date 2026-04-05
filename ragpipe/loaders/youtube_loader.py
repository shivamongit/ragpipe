"""YouTube transcript loader — extract video transcripts as Documents."""

from __future__ import annotations

import re

from ragpipe.core import Document


class YouTubeLoader:
    """Load YouTube video transcripts as Documents.

    Uses youtube-transcript-api to fetch auto-generated or manual captions.

    Usage:
        loader = YouTubeLoader()
        doc = loader.load("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        doc = loader.load("dQw4w9WgXcQ")  # video ID also works
    """

    def __init__(self, languages: list[str] | None = None):
        self.languages = languages or ["en"]

    def _extract_video_id(self, url_or_id: str) -> str:
        """Extract video ID from a URL or return as-is if already an ID."""
        patterns = [
            r'(?:v=|\/v\/|youtu\.be\/)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$',
        ]
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        raise ValueError(f"Could not extract YouTube video ID from: {url_or_id}")

    def load(self, url_or_id: str) -> Document:
        """Load transcript from a YouTube video URL or ID."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "Install youtube-transcript-api: pip install youtube-transcript-api"
            )

        video_id = self._extract_video_id(url_or_id)

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try manual transcripts first, then auto-generated
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(self.languages)
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(self.languages)
            except Exception:
                pass

        if transcript is None:
            # Fallback: get any available transcript
            for t in transcript_list:
                transcript = t
                break

        if transcript is None:
            return Document(
                content="",
                metadata={"source": f"youtube:{video_id}", "loader": "youtube", "error": "No transcript found"},
            )

        entries = transcript.fetch()
        # Build full text with timestamps
        lines = []
        for entry in entries:
            text = entry.get("text", "").strip()
            if text:
                lines.append(text)

        content = " ".join(lines)

        # Calculate total duration
        total_duration = 0
        if entries:
            last = entries[-1]
            total_duration = int(last.get("start", 0) + last.get("duration", 0))

        return Document(
            content=content,
            metadata={
                "source": f"youtube:{video_id}",
                "video_id": video_id,
                "loader": "youtube",
                "language": transcript.language,
                "duration_seconds": total_duration,
                "segment_count": len(entries),
            },
        )
