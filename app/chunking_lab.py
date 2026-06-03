def split_into_chunks(text, chunk_size):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]