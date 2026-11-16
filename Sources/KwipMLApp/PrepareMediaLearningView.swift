import SwiftUI
import KwipMLCore

enum PrepareMediaLessonContent {
    static let objective="Prepare media turns captured pixels or PCM samples into explicitly shaped tensors. The source stays available so you can compare what entered the workflow with what you prepared.\n\nThis canvas has two independent paths:\n- **Image:** a tiny six-color PNG → Prepare media → View.\n- **Audio:** a synthetic 440 Hz PCM tone → Prepare media → View.\n\nThere is no model in this practice. You will inspect inputs that a later model could consume, rather than treating a successful preparation as a prediction."
    static let terms="- **Sample rate:** how many audio samples represent one second.\n- **Channel:** one stream of audio, or one image value such as red.\n- **Axis:** what one tensor dimension represents; axis order changes how a model reads the same values.\n- **Window:** a fixed-length part of one recording.\n- **Valid mask:** whole-number values marking actual samples and padding."
    static let steps=[
        "1. Run Flow.\n2. Select [Read color image](kwip-node://data/source) and [Read PCM WAV](kwip-node://data/audio-source) in turn.\n3. Inspect each captured table.\n\nThe image row has image_content. The audio row has audio_content. File Input preserves the WAV's 8,000 Hz rate, one channel and 16-bit integer samples. Reading it does not resample or normalize it. Audio decoding is an explicit PCM WAV reader choice; other codecs are not supported by this reader.",
        "1. Select [Prepare image pixels](kwip-node://data/image-prepare).\n2. Inspect Fit and pad, width 4, height 4, float32, 0…1 scaling and Channels × height × width.\n3. Open prepared in [Inspect prepared image](kwip-node://data/image-view).\n\nOne observation now has shape 3 × 4 × 4: three color channels, four rows and four columns. Padding creates extra pixel locations; the original captured image is retained. Scaling converts unsigned channel values to decimal values from zero to one. Apply changes before running again.",
        "1. Select [Prepare audio windows](kwip-node://data/audio-prepare).\n2. Keep float32 conversion, the original rate and channel, window length 256, hop 256 and final-window padding.\n3. Inspect [Inspect audio and masks](kwip-node://data/audio-view).\n\nThe 1,000-sample tone creates four rows. Each prepared tensor has shape 1 × 256: one channel and 256 samples. The last row has 232 valid samples and 24 zeros added as padding. Its int64 mask marks valid samples separately from padding; valid length and parent observation are retained as companion columns.",
        "1. Open an audio_content cell and inspect its waveform. Playback starts only when you press Play.\n2. Open a prepared audio window and compare its exact values, shape and source interval.\n3. Try a different hop, Apply, run again and compare row counts.\n\nA smaller hop creates overlapping windows and more observations. It does not create more independent recordings. Partition source recordings before making windows so one recording cannot leak into both Training and Validation.",
        "Before connecting a model, decide its expected dtype, sample dimensions and axis order. In Train's existing Data mapping, choose Prepared tensor and the matching column; required masks are separate explicit inputs. A mismatch stops with a diagnostic rather than silently reshaping or converting data.\n\nA saved model retains its declared input contract. This preparation does not turn it into an automatic raw-file image/audio processor. Use the same explicit preparation recipe for new inputs. For speech-to-text, open Transcribe a private recording: its separate Load model connection supplies the matching processor. For spectral analysis, choose Magnitude, Power or Log-mel in Prepare media; the tensor uses Channels × frames × frequency with an explicit FFT, frame hop and valid-frame mask. This practice adds no automatic microphone or model download."
    ]
    static let boundary="The original pixels/samples and prepared tensors are immutable owned artifacts. View displays captured evidence; display scaling and playback do not change model inputs. Preparation is deterministic and learns no statistics. Apply saves settings; Run Flow performs preparation. Save the lesson as a bookmark or Start fresh to reopen a separate supplied canvas. This editable practice is unscored."
}

struct PrepareMediaLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail {LessonTextPanel(title:"Prepare media · editable practice",text:PrepareMediaLessonContent.objective)}
            Button("Open media preparation practice") {_=store.loadComponentWorkflowExample(PrepareMediaPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
            if !inLessonRail {LessonTextPanel(title:"Try it on the canvas",text:PrepareMediaLessonContent.steps.enumerated().map {"**Step \($0.offset+1)**\n"+$0.element}.joined(separator:"\n\n"))}
        }.accessibilityIdentifier("learning.prepare-media")
    }
}
