import SwiftUI
import KwipMLCore

enum SpeechTranscriptionLessonContent {
    static let objective="This canvas turns a recording into text you can inspect and export. File Input captures your PCM WAV, Prepare media makes mono float32 samples at 16,000 Hz, and a separate Load model provides Whisper's verified weights, tokenizer and audio processor. Transcribe audio combines those two inputs.\n\nNothing is downloaded or recorded automatically. Choose a recording you are allowed to use; this practice intentionally opens without private audio or a model download."
    static let terms="- **Sample rate:** how many samples represent one second.\n- **Chunk:** one bounded part of the recording processed sequentially.\n- **Transcript:** recognized text for one input observation.\n- **Segment:** recognized text with a source-time interval. These are model estimates, not guaranteed word alignment."
    static let steps=[
        "Select [Read your recording](kwip-node://data/recording). Choose a short integer PCM WAV. Keep the explicit PCM WAV reader. File Input preserves the original rate, channels and bit depth; it does not send your recording to a cloud service. Unsupported codecs or oversized files stop with a diagnostic.",
        "Select [Prepare mono waveform](kwip-node://data/waveform). The recipe converts PCM to float32, averages channels to mono and resamples to 16,000 Hz. Keep Samples and whole-item windows for transcription. Generic spectra are an optional analysis path; they do not replace Whisper's matching processor. Partition whole recordings before creating overlapping windows for training.",
        "Select [Load speech model](kwip-node://control/speech-model). Review the pinned openai/whisper-tiny revision, repository license and file inventory. Resolve revision, then Download & verify. Alternatively choose a local folder containing the matching safe weights, tokenizer and processor. This inference adapter does not fine-tune Whisper. Once captured, the model can be used offline.",
        "Select [Transcribe recording](kwip-node://data/speech). Check its waveform column and Model connection. Choose the spoken language or detection per chunk, chunk length, beam width and token allowance. Apply changes, then Run Flow. Cancellation stops processing; it never silently retries a write. Chunk boundaries may split phrases, and a token cap may truncate the text.",
        "Inspect [Inspect transcripts](kwip-node://data/transcripts) and [Inspect source segments](kwip-node://data/segments). Double-click long text for the bounded full-cell viewer. Open a segment's captured audio to review its highlighted interval; playback is opt-in. Compare the text against the recording, particularly silence and names. Speech recognition may hallucinate or omit words. Table Output can export either complete table, and History preserves the captured result according to your retention settings."
    ]
    static let boundary="This editable practice is unscored. Local CPU inference, one recording at a time, with bounded chunks. No microphone, streaming, translation, speaker diarization, speech synthesis or ASR training. Model timestamps and text require human review. Save keeps your project and private model/source bindings; Start fresh opens a separate practice. Portable Code export currently refuses speech-resource workflows explicitly; use native Run Flow and Table Output."
}
struct SpeechTranscriptionLearningView:View {
    @ObservedObject var store:WorkspaceStore
    var body:some View {Button("Open speech transcription practice"){_=store.loadComponentWorkflowExample(SpeechTranscriptionPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)}
}
