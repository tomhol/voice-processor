Here are the best high-quality, free datasets for finding clean English voice samples with annotations:


1. LJSpeech Dataset (https://keithito.com/LJ-Speech-Dataset/) (The Industry Standard)
  This is the most famous single-speaker dataset used for training TTS.
   * Voice: Female, professional narrator (reading non-fiction books).
   * Why use it: Extremely clean, zero background noise, and perfectly punctuated.
   * Access: You can browse the files directly. Each WAV file has a corresponding line of text in metadata.csv.


2. LibriSpeech (ASR corpus) (https://www.openslr.org/12)
  A massive collection of approximately 1,000 hours of 16kHz read English speech from audiobooks.
   * Voice: Hundreds of different male and female speakers of various ages.
   * Why use it: Great for finding "character" voices (older men, younger women, etc.).
   * How to use: Look for the dev-clean or test-clean subsets. Inside, you'll find folders for each speaker, with .txt files containing the transcript for each .flac or .wav file.


3. Common Voice (by Mozilla) (https://commonvoice.mozilla.org/)
  A massive, crowdsourced multi-lingual dataset.
   * Voice: Real people with natural accents (British, American, Australian, Indian, etc.).
   * Why use it: If you want a "real person" sound rather than a "professional narrator" sound.
   * Note: Quality varies, so you have to pick "validated" clips.


4. VCTK Dataset (https://datashare.ed.ac.uk/handle/10283/3443)
  110 English speakers with various accents.
   * Voice: Mostly British accents.
   * Why use it: Each speaker reads the same set of sentences, so you can test how the same script sounds in 110 different voices.

  ---


  Pro-Tip for F5-TTS Reference Clips:
  F5-TTS is sensitive to the reference clip. For the best results:
   1. Length: Use a clip between 5 and 10 seconds. Too short (<3s) and it doesn't "catch" the voice; too long and it can get confused.
   2. Punctuation: Ensure the --ref-text matches the audio exactly, including commas and periods. F5-TTS uses the punctuation in the reference to understand the speaker's rhythm.
   3. Silence: Trim any long silences at the start or end of your reference WAV file.

