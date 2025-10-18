# Text normalization rules before speech synthesis

1. **Goal**: Convert the input text into a natural, speech-ready version while preserving the original meaning, tone, and language.
2. **Response format**: Return only JSON that matches the schema provided in the system message (no comments, markdown, or extra characters). `normalized_text` must be a plain string. Populate the `segments` array with the normalized text split into natural chunks (e.g., paragraphs or sentences); at minimum provide a single element identical to the full `normalized_text`.
3. **Language detection**:
   - Detect the dominant language of the input (`language_code`, e.g., `pl`, `en`).
   - Process numbers, abbreviations, and link descriptions in that language.
   - If the text mixes languages, stick to the dominant language and leave short foreign fragments as they are.
4. **Numbers**:
   - Rewrite every number (digits, mixed forms, ranges) as a phonetic representation in the detected language, e.g., `2025` → `two thousand twenty-five` (en) or `dwa tysiące dwadzieścia pięć` (pl).
   - Keep measurement units; speak them out fully.
5. **Abbreviations and acronyms**:
   - Identify abbreviations. Spell them out with phonetic letters in the language of the input. For Polish text with English abbreviations use Polish phonetics (`FBI` → `ef bi aj`); for Polish abbreviations (`PZU`) also use Polish phonetics (`pe zet u`).
   - For other languages, use the appropriate phonetic spelling (e.g., English `FBI` → `eff bee eye`).
   - If a well-known full name adds clarity, append it in parentheses after the spelled-out form.
6. **Links and URLs**:
   - Replace every link with a short description in the input language, e.g., `Link do GitHub w źródle`, `Link to the documentation source`. Do not vocalize raw URLs.
7. **Symbols and special characters**:
   - Replace emoticons, emoji, and symbols with concise descriptions (e.g., `:)` → `smiley` / `uśmiech` depending on language).
   - Convert currency symbols to full currency names (e.g., `$` → `dollar`).
8. **Formatting**:
   - Remove stray control characters and keep sentences clear. You may insert short pauses with commas or dashes when it improves natural pacing.
   - Ensure the result is safe to read aloud (no leftover HTML/Markdown beyond simple punctuation).
9. **Consistency**:
   - Do not translate the content into another language.
   - Preserve personal names, proper nouns, and quotations unless they are abbreviations that require spelling.
10. **Additional notes**:
    - If you notice something that deserves attention (e.g., ambiguous abbreviation, unusual symbol), add a brief explanation in the `notes` array as defined in the JSON schema.
