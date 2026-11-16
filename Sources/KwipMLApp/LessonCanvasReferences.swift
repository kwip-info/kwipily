import Foundation
import KwipMLCore

/// Authored addresses for the supplied canvases. Never infer a target from a label or prose.
/// Existing per-step author references win; presentation-only changes preserve reading revisions.
@MainActor enum LessonCanvasReferences {
    private static func d(_ id:String)->WorkflowNodeAddress {.init(domain:.data,localID:id)}
    private static func c(_ id:String)->WorkflowNodeAddress {.init(domain:.control,localID:id)}
    private static func m(_ id:String)->WorkflowNodeAddress {.init(domain:.model,localID:id)}
    static let addresses:[String:[String:[WorkflowNodeAddress]]] = [
        "prepare-media":["goal":[d("source"),d("audio-source")],"try.0":[d("source"),d("audio-source")],"try.1":[d("image-prepare"),d("image-view")],"try.2":[d("audio-prepare"),d("audio-view")],"try.3":[d("audio-source"),d("audio-prepare"),d("audio-view")],"try.4":[d("image-prepare"),d("audio-prepare")],"inspect":[d("image-view"),d("audio-view")]],
        "document-classification":["goal":[d("documents"),c("classifier"),c("predict")],"try.0":[d("documents"),d("labels"),d("join"),c("classifier")],"try.1":[d("partition")],"try.2":[d("features")],"try.3":[c("classifier")],"try.4":[c("save"),c("load"),d("unseen"),c("predict"),d("predictions")],"inspect":[c("classifier"),c("predict")]],
        "extract-text":["goal":[d("scans"),d("pixels"),d("ocr")],"try.0":[d("ocr")],"try.1":[d("scans"),d("pixels")],"try.2":[d("ocr")],"try.3":[d("regions")],"try.4":[d("documents"),d("regions")],"try.5":[d("scans"),d("ocr")],"inspect":[d("documents"),d("regions")]],
        "transcribe-audio":["goal":[d("recording"),d("waveform"),c("speech-model"),d("speech")],"try.0":[d("recording")],"try.1":[d("waveform")],"try.2":[c("speech-model")],"try.3":[d("speech")],"try.4":[d("transcripts"),d("segments")],"inspect":[d("speech")]],
        "classify-audio":["goal":[d("recording"),d("waveform"),c("classifier-model"),d("classifier")],"try.0":[d("recording")],"try.1":[d("waveform")],"try.2":[c("classifier-model")],"try.3":[d("classifier")],"try.4":[d("predictions"),d("scores")],"inspect":[d("classifier")]],
        "game-rules":["goal":[d("rules"),d("tests"),d("trace")],"try.0":[d("rules")],"try.1":[d("rules")],"try.2":[d("rules")],"try.3":[d("rules")],"try.4":[d("tests"),d("trace")],"try.5":[d("rules")],"inspect":[d("rules")]],
        "game-episodes":["goal":[d("rules"),m("observation"),m("policy-head"),c("session")],"try.0":[d("rules")],"try.1":[m("observation"),m("policy-head"),m("output")],"try.2":[c("session")],"try.3":[d("episodes")],"try.4":[d("moves")],"try.5":[c("session"),d("episodes")],"inspect":[c("session")]],
        "self-play":["intro":[d("rules"),m("shared"),m("policy"),m("value"),c("train")],"heads":[m("observation"),m("output"),m("policy"),m("value")],"configure":[c("train")],"inspect":[c("train"),d("iterations"),d("evaluation")],"campaign":[c("train")],"compare":[c("train"),d("evaluation")],"reuse":[c("session"),c("save")]],
        "prebuilt-models":["goal":[d("customers"),c("segments")],"try.0":[d("customers"),d("numeric")],"try.1":[c("segments")],"try.2":[d("assignments")],"try.3":[d("profiles")],"try.4":[c("save"),c("segments")],"try.5":[c("segments")],"inspect":[c("segments")]],
        "typed-decisions":["goal":[d("tickets"),c("decision-model"),c("train"),d("decide")],"try.0":[d("tickets"),d("partition")],"try.1":[c("decision-model")],"try.2":[c("train")],"try.3":[c("save-model"),c("train")],"try.4":[d("decisions"),d("scores")],"try.5":[c("save-model"),d("decide")],"inspect":[d("decide")]],
        "mixture-experts":["goal":[m("input"),m("mixture"),m("output")],"open":[m("input"),m("mixture"),m("output")],"try.0":[m("mixture")],"try.1":[m("mixture")],"try.2":[m("mixture"),c("train")],"try.3":[c("train")],"try.4":[c("train")],"try.5":[c("train")],"inspect":[m("mixture"),c("train")]],
        "fine-tuning":["goal":[d("observations"),c("loaded-model"),c("train")],"open":[c("loaded-model"),d("partition")],"try.0":[c("train")],"try.1":[c("train")],"try.2":[c("train")],"try.3":[c("train")],"try.4":[c("save-model"),c("train"),c("generate")],"try.5":[c("train")],"try.6":[c("save-model"),c("train")],"inspect":[c("train"),c("save-model")]]
    ]
    static func annotate(_ content:LessonContent)->LessonContent {
        guard let steps=addresses[content.entryID] else{return content}
        var result=content
        result.steps=result.steps.map {step in
            guard step.nodeFocus == nil,let references=steps[step.id] else{return step}
            var updated=step;updated.nodeFocus = .init(addresses:references);return updated
        }
        return result
    }
}
