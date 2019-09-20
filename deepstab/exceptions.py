class NonCoding(Exception):
    """Called when a transcript is being annotated but has no CDS. We are not interested in these transcripts."""
    pass


