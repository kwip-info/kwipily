import SwiftUI
import KwipMLCore

struct OCRConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var error=""
    private var accent:Color {ComponentVisualFamily.data.primary}
    private var pending:Bool {draft != original}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Source",symbol:"photo",identifier:"ocr.source",initiallyExpanded:true,accent:accent) {
                let schema=context.upstreamSchema(forInput:"table")
                let columns=schema.filter {$0.dtype=="image" || $0.dtype=="tensor"}
                if columns.isEmpty {text("Image column",key:"inputColumn")} else {choice("Image column",key:"inputColumn",values:columns.map{($0.name,$0.name)})}
                let groups=schema.filter {$0.dtype == "string"}
                let ordering=schema.filter {$0.dtype == "int64"}
                if schema.isEmpty {
                    text("Document grouping column · blank for each row",key:"groupBy")
                    text("Page/frame ordering column",key:"pageColumn")
                } else {
                    choice("Document grouping column",key:"groupBy",values:[("","Each row")]+groups.map{($0.name,$0.name)})
                    choice("Page/frame ordering column",key:"pageColumn",values:[("","Source row order")]+ordering.map{($0.name,$0.name)})
                }
            }
            InspectorSection("OCR model",symbol:"shippingbox",identifier:"ocr.model",initiallyExpanded:true,accent:accent) {
                Text("English printed text · CPU").fontWeight(.semibold)
                Button("Review model pack…") { ApplicationNavigationWindows.shared.showSettings(page: .models) }
                    .buttonStyle(FlatButtonStyle())
                Text("Optional pack · available offline after download").foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Recognition",symbol:"text.viewfinder",identifier:"ocr.recognition",initiallyExpanded:true,accent:accent) {
                choice("Decoder",key:"decoder",values:[("greedy","Greedy"),("beamsearch","Beam search")])
                if draft["decoder"] == .string("beamsearch") {entry("Beam width",key:"beamWidth",integer:true)}
                entry("Review below confidence",key:"reviewThreshold")
                VStack(alignment:.leading,spacing:4) {
                    Text("Additional recognition rotations · degrees").foregroundStyle(MLTheme.muted)
                    FileInputDraftEntry(title:"Rotation candidates · 90, 180, 270",value:binding("rotationCandidates"),format:FileInputDraftText.display,parse:{raw in
                    let values=raw.split(separator:",").map {$0.trimmingCharacters(in:.whitespaces)}
                    if values.isEmpty {return .array([])}
                    guard values.allSatisfy({Int64($0) != nil}) else {return .string(raw)}
                    return .array(values.compactMap{Int64($0).map(JSONValue.integer)})
                    }).help("Optional comma-separated 90, 180 or 270 degree recognition retries. Blank keeps the detected orientation.")
                }
                text("Character allowlist",key:"allowlist");text("Character blocklist",key:"blocklist")
            }
            InspectorSection("Detection & grouping",symbol:"rectangle.dashed",identifier:"ocr.detection",initiallyExpanded:false,accent:accent) {
                entry("Text threshold",key:"textThreshold");entry("Low-text threshold",key:"lowText");entry("Link threshold",key:"linkThreshold")
                entry("Minimum region size · pixels",key:"minimumSize",integer:true)
                entry("Detection maximum side · pixels",key:"canvasSize",integer:true);entry("Detection magnification",key:"magnification")
                entry("Line width threshold",key:"widthThreshold");entry("Slope threshold",key:"slopeThreshold")
                entry("Line center threshold",key:"ycenterThreshold");entry("Line height threshold",key:"heightThreshold");entry("Box margin",key:"margin")
            }
            InspectorSection("Processing limits",symbol:"gauge",identifier:"ocr.limits",initiallyExpanded:false,accent:accent) {
                entry("Maximum regions · complete run",key:"maximumRegions",integer:true)
                entry("Maximum text characters · complete run",key:"maximumCharacters",integer:true)
            }
            InspectorSection("Output",symbol:"tablecells",identifier:"ocr.output",initiallyExpanded:true,accent:accent) {
                text("Documents text column",key:"outputColumn")
                Text("Documents + Regions → standard tables").foregroundStyle(MLTheme.muted)
            }
            HStack {Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!pending);Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)}
            if !error.isEmpty {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).onAppear(perform:reload).onChange(of:node.id){_,_ in reload()}.onChange(of:draft){_,_ in publish()}
            .onChange(of:node.config){_,next in if !pending {reload()} else if next != original {error="Saved settings changed. Revert to load them; the draft is preserved."}}
            .onDisappear{context.setNumericPending(origin:context.numericOrigin(key:"ocr"),key:"ocr",pending:false)}
    }
    private func binding(_ key:String)->Binding<JSONValue> {Binding(get:{draft[key] ?? .string("")},set:{draft[key]=$0})}
    private func text(_ title:String,key:String)->some View {VStack(alignment:.leading,spacing:4){Text(title).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})).flatField()}}
    private func entry(_ title:String,key:String,integer:Bool=false)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).foregroundStyle(MLTheme.muted)
            FileInputDraftEntry(title:title,value:binding(key),format:FileInputDraftText.display,parse:{raw in
            if integer,let n=Int64(raw) {return .integer(n)}
            if !integer,let n=Double(raw),n.isFinite {return .number(n)}
            return .string(raw)
            }).help(fieldHelp(key))
        }
    }
    private func choice(_ title:String,key:String,values:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:title,selection:draft[key]?.stringValue ?? "",options:values.map{.init(id:$0.0,title:$0.1)}){draft[key] = .string($0)}
                .help(fieldHelp(key))
        }
    }
    private func fieldHelp(_ key:String)->String {
        switch key {
        case "canvasSize":return "Maximum image side used for detection after magnification. Raising this alone does not enlarge a smaller image or add source detail."
        case "magnification":return "Enlarge the detector input up to the maximum side. Recognition still reads the source image; use a higher-resolution original when possible."
        case "reviewThreshold":return "Flag regions below this model-confidence score for review. This does not change recognition, remove text or measure accuracy."
        case "decoder":return "Greedy selects the most likely character sequence. Beam search considers alternatives and costs more CPU; it cannot recover missing image detail."
        case "minimumSize":return "Ignore detected boxes smaller than this pixel size. Small characters may be missed when the source image is reduced."
        case "textThreshold":return "Detector score required for text regions. Lowering it may find faint text but also introduce false detections."
        case "lowText":return "Lower detector score used to connect candidate text pixels."
        case "linkThreshold":return "Detector score used to connect adjacent character regions."
        case "widthThreshold","ycenterThreshold","heightThreshold","margin":return "Box grouping or margin distance, measured relative to text-box height. Changes can join neighboring table cells; inspect Regions."
        case "slopeThreshold":return "Maximum slope allowed when merging nearby boxes into a line."
        case "maximumRegions","maximumCharacters":return "Complete-run safety allowance. Exceeding it fails explicitly; it is not a preview size or silent truncation."
        default:return ""
        }
    }
    private func publish(){context.setNumericPending(origin:context.numericOrigin(key:"ocr"),key:"ocr",pending:pending)}
    private func reload(){draft=node.config;original=node.config;error="";publish()}
    private func apply(){do {guard let definition=context.operatorDefinition else {return};try OCRConfiguration.validate(draft,definition:definition);if context.applyTableConfiguration(draft,expected:original){original=draft;error="";publish()}else{error=context.statusMessage}}catch{self.error=error.localizedDescription}}
}
