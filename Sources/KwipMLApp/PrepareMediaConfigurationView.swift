import SwiftUI
import KwipMLCore

struct PrepareMediaConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var origin=""
    @State private var failure:String?
    @State private var pendingFields=Set<String>()
    @State private var generation=UUID()
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var spectral:Bool {draft["kind"] == .string("audio") && draft["representation"] != nil && draft["representation"] != .string("samples")}
    private var audio:Bool {draft["kind"] == .string("audio")}
    private var pending:Bool {draft != original || !pendingFields.isEmpty}
    private var schema:[DataColumnSchema] {context.upstreamSchema(forInput:"table")}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Source",symbol:"tablecells",identifier:"prepare-media.source",initiallyExpanded:true,accent:accent) {
                choice("Media",key:"kind",options:[("image","Image"),("audio","PCM WAV audio")])
                let options=schema.filter {$0.dtype == (audio ? "audio":"image") || audio && $0.dtype=="tensor"}.map {($0.name,$0.name)}
                if options.isEmpty {text("Captured media column",key:"inputColumn")}
                else {choice("Captured media column",key:"inputColumn",options:options)}
                Label("Original media is retained",systemImage:"checkmark.shield").foregroundStyle(accent)
            }
            if audio {audioFields} else {imageFields}
            InspectorSection("Output",symbol:"square.stack.3d.up",identifier:"prepare-media.output",initiallyExpanded:true,accent:accent) {
                if !audio {choice("Output type",key:"outputMode",options:[("tensor","Prepared tensor"),("image","Image for OCR")])}
                text("Prepared column",key:"outputColumn")
                Text(audio ? (spectral ? "Channels × frames × frequency · one row per window":"Channels × samples · one row per window") : (draft["layout"] == .string("CHW") ? "Channels × height × width":"Height × width × channels")).foregroundStyle(MLTheme.muted)
                if audio {
                    let name=draft["outputColumn"]?.stringValue ?? "prepared"
                    Text(name+"_mask · int64 valid-"+(spectral ? "frame":"sample")+" mask\n"+name+(spectral ? "_valid_frames":"_valid_samples")+" · valid length\n"+name+"_parent · original observation").font(.system(size:11,design:.monospaced)).textSelection(.enabled)
                    Text("Partition source files before making windows. Windows from the same recording must stay in one partition.").foregroundStyle(MLTheme.muted)
                }
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!pending || !pendingFields.isEmpty)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer();if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied media preparation changes")}
            }
            if let failure {Text(failure).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).accessibilityIdentifier("prepare-media.configuration")
            .onAppear(perform:reload).onChange(of:node.id){_,_ in reload()}
            .onChange(of:draft){_,_ in publish()}
            .onChange(of:node.config){_,next in if !pending {reload()} else if next != original {failure="Saved settings changed. Revert to load them; your draft is preserved."}}
            .onDisappear {context.setNumericPending(origin:origin,key:"prepareMedia",pending:false)}
    }
    private var imageFields:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Geometry",symbol:"crop",identifier:"prepare-media.geometry",initiallyExpanded:true,accent:accent) {
                Toggle("Crop source region",isOn:Binding(get:{draft["cropEnabled"] == .bool(true)},set:{draft["cropEnabled"] = .bool($0)}))
                if draft["cropEnabled"] == .bool(true) {number("Crop left · pixels",key:"cropX");number("Crop top · pixels",key:"cropY");number("Crop width",key:"cropWidth");number("Crop height",key:"cropHeight")}
                FlatChoiceControl(title:"Clockwise rotation",selection:String(draft["rotation"]?.integerValue ?? 0),options:[0,90,180,270].map{.init(id:String($0),title:String($0)+"°")}){if let n=Int64($0){draft["rotation"] = .integer(n)}}
                choice("Size",key:"sizeMode",options:[("preserve","Preserve original"),("resize","Resize to dimensions"),("fitPad","Fit and pad"),("crop","Fill and center crop")])
                if draft["sizeMode"] != .string("preserve") {
                    number("Width · pixels",key:"width");number("Height · pixels",key:"height")
                    choice("Interpolation",key:"interpolation",options:[("nearest","Nearest neighbor"),("bilinear","Bilinear"),("bicubic","Bicubic")])
                    Text("Fit uses zero padding. Source coordinates are retained as a pixel-center transform.").foregroundStyle(MLTheme.muted)
                }
            }
            InspectorSection("Color & values",symbol:"circle.lefthalf.filled",identifier:"prepare-media.values",initiallyExpanded:true,accent:accent) {
                choice("Color",key:"color",options:[("preserve","Preserve source channels"),("RGB","RGB"),("L","Grayscale")])
                choice("Alpha",key:"alpha",options:[("preserve","Preserve alpha"),("drop","Drop alpha explicitly"),("composite","Composite onto matte")])
                if draft["alpha"] == .string("composite") {ForEach(0..<3,id:\.self) {i in matte(i)}}
                choice("Sample values",key:"sampleFormat",options:[("preserve","Preserve source precision"),("float32","Convert to float32")])
                if draft["sampleFormat"] == .string("float32") {choice("Scaling",key:"scaling",options:[("none","Preserve numeric range"),("unit","Unsigned pixels to 0…1")])}
                choice("Tensor layout",key:"layout",options:[("HWC","Height × width × channels"),("CHW","Channels × height × width")])
            }
        }
    }
    private var audioFields:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Channels & rate",symbol:"waveform",identifier:"prepare-media.audio",initiallyExpanded:true,accent:accent) {
                choice("Sample values",key:"sampleFormat",options:[("preserve","Preserve integer PCM"),("float32","Normalize PCM to float32")])
                choice("Channels",key:"channels",options:[("preserve","Preserve all channels"),("select","Select one channel"),("mono","Average to mono")])
                if draft["channels"] == .string("select") {number("Channel index · starts at 0",key:"channelIndex")}
                number("Sample rate · 0 preserves original",key:"targetRate")
                Text("Resampling and mono mixing require float32. PCM normalization uses its original bit depth; no peak normalization is fitted.").foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Windows",symbol:"rectangle.split.3x1",identifier:"prepare-media.windows",initiallyExpanded:true,accent:accent) {
                number("Window length · samples · 0 keeps whole item",key:"windowSamples")
                choice("Mask meaning",key:"maskConvention",options:[("valid","1 = valid sample · 0 = padding"),("blocked","1 = padding / blocked · 0 = valid")])
                if draft["windowSamples"] != .integer(0) {
                    number("Hop · samples · 0 uses window length",key:"hopSamples")
                    choice("Incomplete final window",key:"tail",options:[("drop","Drop incomplete window"),("pad","Pad with zeros and retain valid mask")])
                }
            }
            InspectorSection("Representation",symbol:"waveform.path",identifier:"prepare-media.spectral",initiallyExpanded:true,accent:accent) {
                choice("Prepared values",key:"representation",options:[("samples","Samples"),("magnitude","Magnitude spectrum"),("power","Power spectrum"),("logMel","Log-mel power · dB")])
                if spectral {
                    number("FFT size · samples",key:"fftSize");number("Frame hop · samples",key:"spectralHop")
                    choice("Window function",key:"window",options:[("hann","Periodic Hann"),("hamming","Periodic Hamming"),("rectangular","Rectangular")])
                    Toggle("Center frames · zero pad edges",isOn:Binding(get:{draft["center"] != .bool(false)},set:{draft["center"] = .bool($0)}))
                    if draft["representation"] == .string("logMel") {
                        number("Mel bands · HTK scale · unit peak",key:"melBands");number("Minimum frequency · Hz",key:"minimumHz");number("Maximum frequency · 0 uses Nyquist",key:"maximumHz");number("dB floor · reference power 1",key:"logFloorDB")
                    }
                    Text("Channels × frames × frequency. Requires float32. This generic analysis does not replace a speech model's matching processor.").foregroundStyle(MLTheme.muted)
                }
            }
            InspectorSection("Resampling filter",symbol:"slider.horizontal.3",identifier:"prepare-media.filter",initiallyExpanded:false,accent:accent) {
                number("Windowed-sinc filter radius",key:"filterRadius")
                Text("Rational-phase low-pass filtering with a Hann window and zero boundary extension. Rate ratio and filter work are bounded before allocation.").foregroundStyle(MLTheme.muted)
            }
        }
    }
    private func choice(_ title:String,key:String,options:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:title,selection:draft[key]?.stringValue ?? (try? PrepareMediaConfiguration.catalog().defaults[key]?.stringValue) ?? "",options:options.map {.init(id:$0.0,title:$0.1)},accent:accent) {value in
                if key=="kind",value != draft[key]?.stringValue {draft=(try? PrepareMediaConfiguration.defaults(kind:value)) ?? draft}
                else {draft[key] = .string(value)}
                if key=="sampleFormat",value=="preserve" {draft["scaling"] = .string("none")}
            }.accessibilityIdentifier("prepare-media."+key)
        }
    }
    private func text(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})).flatField()}.accessibilityIdentifier("prepare-media."+key)
    }
    private func number(_ title:String,key:String)->some View {
        let bounds=(try? PrepareMediaConfiguration.catalog().integerBounds[key]) ?? [0,1]
        return VStack(alignment:.leading,spacing:4) {
            Text(title).foregroundStyle(MLTheme.muted)
            BoundedNumberField(title:title,value:.wholeDisplay(Double(draft[key]?.integerValue ?? (try? PrepareMediaConfiguration.catalog().defaults[key]?.integerValue) ?? bounds[0])),domain:.integer(minimum:Double(bounds[0]),maximum:Double(bounds[1])),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{origin,pending in
                if pending {pendingFields.insert(key)} else {pendingFields.remove(key)}
                context.setNumericPending(origin:origin,key:key,pending:pending)
            }) {value in guard case .integer(let n)=value else {return false};draft[key] = .integer(n);if key=="windowSamples",n==0 {draft["hopSamples"] = 0};return true}.id(generation)
        }.accessibilityIdentifier("prepare-media."+key)
    }
    private func matte(_ index:Int)->some View {
        let name=["Red","Green","Blue"][index],key="matte"+String(index)
        return VStack(alignment:.leading,spacing:4) {
            Text(name+" matte · 0…1").foregroundStyle(MLTheme.muted)
            FileInputDraftEntry(title:name+" matte",value:Binding(get:{if case .array(let values)?=draft["matte"] {return values[index]};return .integer(0)},set:{value in if case .array(var values)?=draft["matte"] {values[index]=value;draft["matte"] = .array(values)}}),format:FileInputDraftText.display,parse:{raw in if let n=Double(raw),n.isFinite {return .number(n)};return .string(raw)})
                .accessibilityIdentifier("prepare-media."+key)
        }
    }
    private func publish(){context.setNumericPending(origin:origin,key:"prepareMedia",pending:pending)}
    private func reload(){generation=UUID();pendingFields=[];context.setNumericPending(origin:origin,key:"prepareMedia",pending:false);draft=node.config;original=node.config;origin=context.numericOrigin(key:"prepareMedia");failure=nil;publish()}
    private func apply(){do {try PrepareMediaConfiguration.validate(draft);if context.applyTableConfiguration(draft,expected:original) {original=draft;failure=nil;publish()} else {failure=context.statusMessage}} catch {failure=error.localizedDescription}}
}
