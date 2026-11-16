import SwiftUI
import KwipMLCore

enum AudioClassificationLessonContent {
    static let objective="This canvas identifies sounds in a private recording. File Input captures a PCM WAV, Prepare media makes mono float32 samples at 16,000 Hz, and Load model supplies verified AST weights, its matching audio processor and class names. Classify audio scores each bounded window and sends ordinary tables to View.\n\nMultiple sounds can be present together. This is sound classification, not speech transcription. Nothing downloads or records automatically; choose a recording you have permission to use."
    static let terms="- **Window:** one bounded interval the model examines.\n- **Hop:** the distance between window starts; a shorter hop creates overlap.\n- **Class:** one sound label in the model's fixed vocabulary.\n- **Independent score:** a sigmoid value for one class; all class scores need not sum to one.\n- **Reporting threshold:** controls which labels are flagged, while Scores still retains every class."
    static let steps=[
        "Select [Read your recording](kwip-node://data/recording). Choose a short integer PCM WAV. File Input captures the recording without sending it to a cloud service. Unsupported codecs and oversized files stop with a diagnostic.",
        "Select [Prepare mono waveform](kwip-node://data/waveform). Keep Samples, float32, mono and 16,000 Hz. These settings supply the waveform AST expects; generic log-mel analysis is a different representation and cannot replace its matching processor. Keep actual source lengths and offsets when padding.",
        "Select [Load audio classifier](kwip-node://control/classifier-model). Review the pinned MIT AudioSet AST snapshot, license and download inventory, then Resolve revision and Download & verify. A compatible local folder works too: safe weights, config, class labels and the matching processor are required. No tokenizer or remote code is used. Captured models run offline. This adapter does not train AST.",
        "Select [Classify recording](kwip-node://data/classifier). Confirm its waveform column and Model connection. The default 10-second windows advance by 8 seconds. Choose padding or refusal for a short final window, a reporting threshold and a label limit. Apply changes and Run Flow. Window and complete score-row allowances refuse excess work; they never silently sample away classes.",
        "Open [Inspect window predictions](kwip-node://data/predictions) and [Inspect all class scores](kwip-node://data/scores). Predictions preserves source columns and the top label per window. Scores has every class, rank, score and reporting flags. Double-click captured audio to listen; compare labels against the actual recording. These are coarse window times, not exact event boundaries or calibrated confidence. Table Output exports complete tables; History follows your capture and retention choices."
    ]
    static let boundary="This editable practice is unscored. Local CPU multi-label AST inference, with explicit windows and complete bounded class tables. No microphone, live streaming, classification fine-tuning or arbitrary architecture support. Scores are not a safety or medical judgment. Save retains project/resource bindings; Start fresh opens a separate practice. Portable Code export explicitly refuses model-resource workflows; use native Run Flow and Table Output."
}
struct AudioClassificationLearningView:View {
    @ObservedObject var store:WorkspaceStore
    var body:some View {Button("Open audio classification practice"){_=store.loadComponentWorkflowExample(AudioClassificationPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)}
}
