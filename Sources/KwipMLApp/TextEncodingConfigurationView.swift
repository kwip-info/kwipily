import SwiftUI
import AppKit
import UniformTypeIdentifiers
import KwipMLCore

struct TextEncodingConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var origin=""
    @State private var failure:String?
    @State private var pendingFields=Set<String>()
    @State private var fieldGeneration=UUID()
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var pending:Bool {draft != original || !pendingFields.isEmpty}
    private var documentFeatures:Bool {draft["mode"] == .string("documentFeatures")}
    private var features:[String:JSONValue] {if case .object(let c)=draft["featureSettings"] {return c};return DocumentClassificationConfiguration.featureDefaults}
    private var supplied:Bool {draft["vocabularySource"] == .string("supplied")}
    private var schema:[DataColumnSchema] {context.upstreamSchema(forInput:supplied ? "table":"training")}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            VStack(alignment:.leading,spacing:4) {
                Text("Representation").foregroundStyle(MLTheme.muted)
                FlatChoiceControl(title:"Text representation",selection:documentFeatures ? "documentFeatures":"tokens",options:[.init(id:"tokens",title:"Token IDs"),.init(id:"documentFeatures",title:"Document features · TF-IDF")],accent:accent) {mode in
                    if mode=="documentFeatures" {draft["mode"] = .string(mode);draft["featureSettings"] = .object(DocumentClassificationConfiguration.featureDefaults);draft["vocabularySource"] = .string("training");draft["vocabulary"] = .array([]);draft["keepSourceText"] = true;draft["outputColumn"] = .string("features")}
                    else {draft.removeValue(forKey:"mode");draft.removeValue(forKey:"featureSettings");draft["outputColumn"] = .string("tokens")}
                }
            }
            if documentFeatures {documentFeatureControls} else {
            InspectorSection("Text",symbol:"text.alignleft",identifier:"text-encoding.text",initiallyExpanded:true,accent:accent) {
                choice("Text column",key:"textColumn",options:schema.filter {$0.dtype=="string"}.map {($0.name,$0.name)})
                if schema.isEmpty {text("Exact text column",key:"textColumn")}
                choice("Tokenizer",key:"tokenizer",options:[("character","Character"),("word","Word + whitespace")])
                choice("Unicode normalization",key:"unicodeNormalization",options:[("none","None"),("NFC","NFC"),("NFKC","NFKC")])
                choice("Letter case",key:"letterCase",options:[("preserve","Preserve"),("lower","Lowercase"),("casefold","Case fold")])
                choice("Whitespace",key:"whitespace",options:[("preserve","Preserve"),("trim","Trim ends"),("collapse","Collapse runs")])
                choice("Missing text",key:"missingText",options:[("error","Refuse"),("empty","Empty text")])
            }
            InspectorSection("Vocabulary",symbol:"list.number",identifier:"text-encoding.vocabulary",initiallyExpanded:true,accent:accent) {
                choice("Vocabulary source",key:"vocabularySource",options:[("training","Fit on Training"),("supplied","Supplied vocabulary")])
                if supplied {
                    HStack {Text("\(vocabularyCount) supplied tokens").foregroundStyle(accent);Spacer();Button("Load JSON…",action:loadVocabulary).buttonStyle(FlatButtonStyle())}
                } else {
                    number("Minimum frequency",key:"minimumFrequency",range:1...1000000)
                    choice("ID order",key:"idOrder",options:[("frequency","Frequency, then token"),("lexicographic","Token order")])
                }
                number("Vocabulary limit",key:"vocabularyLimit",range:1...65536)
                choice("Unknown token",key:"unknownPolicy",options:[("unk","Use <unk>"),("error","Refuse")])
            }
            InspectorSection("Special tokens",symbol:"number.square",identifier:"text-encoding.special",initiallyExpanded:false,accent:accent) {
                check("Reserve <pad>",key:"pad")
                check("Add <bos> at the beginning",key:"bos")
                check("Add <eos> at the end",key:"eos")
            }
            InspectorSection("Output",symbol:"tablecells",identifier:"text-encoding.output",initiallyExpanded:true,accent:accent) {
                text("Token column",key:"outputColumn")
                check("Keep source text",key:"keepSourceText")
            }
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!pending || !pendingFields.isEmpty)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer();if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Text encoding changes")}
            }
            if let failure {Text(failure).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).accessibilityIdentifier("text-encoding.configuration")
            .onAppear(perform:reload).onChange(of:node.id){_,_ in reload()}
            .onChange(of:draft){_,_ in publish()}
            .onChange(of:node.config){_,next in if !pending {reload()} else if next != original {failure="Saved settings changed. Revert to load them; your draft is preserved."}}
            .onDisappear {context.setNumericPending(origin:origin,key:"textEncoding",pending:false)}
    }
    private var documentFeatureControls:some View {
        VStack(alignment:.leading,spacing:10) {
            InspectorSection("Document text",symbol:"doc.text",identifier:"text-encoding.documents",initiallyExpanded:true,accent:accent) {
                choice("Text column",key:"textColumn",options:schema.filter {$0.dtype=="string"}.map {($0.name,$0.name)})
                if schema.isEmpty {text("Exact text column",key:"textColumn")}
                Text("Fit vocabulary and IDF on Training only. Validation and Test reuse frozen preparation. Keep whole documents and related groups together in Partition.").foregroundStyle(MLTheme.muted)
            }
            InspectorSection("TF-IDF features",symbol:"textformat.abc",identifier:"text-encoding.tfidf",initiallyExpanded:true,accent:accent) {
                featureChoice("Analyzer",key:"analyzer",options:[("word","Word"),("char","Character")])
                featureNumber("Minimum n-gram",key:"ngramMin",range:1...5)
                featureNumber("Maximum n-gram",key:"ngramMax",range:1...5)
                featureNumber("Vocabulary cap",key:"vocabularyLimit",range:1...65536)
                featureNumber("Minimum document frequency",key:"minimumDF",range:1...1000000)
                featureChoice("Normalization",key:"normalization",options:[("l2","L2"),("l1","L1"),("none","None")])
                featureCheck("Lowercase",key:"lowercase")
            }
            InspectorSection("Advanced",symbol:"slider.horizontal.3",identifier:"text-encoding.tfidf-advanced",initiallyExpanded:false,accent:accent) {
                featureChoice("Accent handling",key:"stripAccents",options:[("none","Preserve"),("unicode","Remove Unicode accents"),("ascii","ASCII transliteration")])
                VStack(alignment:.leading,spacing:4) {Text("Maximum document frequency (proportion)").foregroundStyle(MLTheme.muted)
                    BoundedNumberField(title:"Maximum document frequency",value:.decimal(features["maximumDF"]?.doubleValue ?? 1),domain:.decimal(minimum:0.000001,maximum:1),origin:context.numericOrigin(key:"maximumDF"),currentOrigin:{context.currentNumericOrigin(key:"maximumDF")},onDraftChange:{o,p in context.setNumericPending(origin:o,key:"maximumDF",pending:p);if p {pendingFields.insert("maximumDF")} else {pendingFields.remove("maximumDF")}}) {v in self.setFeature("maximumDF",.number(v.double));return true}
                }
                featureCheck("Binary term counts",key:"binary");featureCheck("Sublinear term frequency",key:"sublinearTF")
                featureCheck("Use inverse document frequency",key:"useIDF");featureCheck("Smooth IDF",key:"smoothIDF")
            }
            InspectorSection("Output",symbol:"tablecells",identifier:"text-encoding.feature-output",initiallyExpanded:true,accent:accent) {
                text("Feature summary prefix",key:"outputColumn")
                Text("Source text and document identity stay intact. The summary counts nonzero terms; Estimator constructs sparse features in the worker. The vocabulary output lists terms and frozen IDF weights.").foregroundStyle(MLTheme.muted)
            }
        }
    }
    private func setFeature(_ key:String,_ value:JSONValue) {var next=features;next[key]=value;draft["featureSettings"] = .object(next)}
    private func featureChoice(_ title:String,key:String,options:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(MLTheme.muted);FlatChoiceControl(title:title,selection:features[key]?.stringValue ?? "",options:options.map{.init(id:$0.0,title:$0.1)},accent:accent){setFeature(key,.string($0))}}
    }
    private func featureCheck(_ title:String,key:String)->some View {Toggle(title,isOn:Binding(get:{features[key] == true},set:{setFeature(key,.bool($0))})).toggleStyle(.checkbox)}
    private func featureNumber(_ title:String,key:String,range:ClosedRange<Int>)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(MLTheme.muted)
            BoundedNumberField(title:title,value:.wholeDisplay(Double(features[key]?.integerValue ?? range.lowerBound)),domain:.integer(minimum:Double(range.lowerBound),maximum:Double(range.upperBound)),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{o,p in context.setNumericPending(origin:o,key:key,pending:p);if p {pendingFields.insert(key)} else {pendingFields.remove(key)}}) {v in guard case .integer(let n)=v else {return false};setFeature(key,.integer(n));return true}.id(fieldGeneration)
        }
    }
    private var vocabularyCount:Int {if case .array(let tokens)=draft["vocabulary"] {return tokens.count};return 0}
    private func choice(_ title:String,key:String,options:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:4) {
        Text(title).foregroundStyle(MLTheme.muted)
        FlatChoiceControl(title:title,selection:draft[key]?.stringValue ?? "",options:options.map {.init(id:$0.0,title:$0.1)},accent:accent) {value in
            draft[key] = .string(value)
            if key=="vocabularySource",value=="training" {draft["vocabulary"] = .array([])}
        }.accessibilityIdentifier("text-encoding."+key)
        }
    }
    private func text(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})).flatField()}.accessibilityIdentifier("text-encoding."+key)
    }
    private func number(_ title:String,key:String,range:ClosedRange<Int>)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).foregroundStyle(MLTheme.muted)
            BoundedNumberField(title:title,value:.wholeDisplay(Double(draft[key]?.integerValue ?? range.lowerBound)),domain:.integer(minimum:Double(range.lowerBound),maximum:Double(range.upperBound)),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{origin,pending in
                if pending {pendingFields.insert(key)} else {pendingFields.remove(key)}
                context.setNumericPending(origin:origin,key:key,pending:pending)
            }) {value in guard case .integer(let n)=value else {return false};draft[key] = .integer(n);return true}.id(fieldGeneration)
        }.accessibilityIdentifier("text-encoding."+key)
    }
    private func check(_ title:String,key:String)->some View {Toggle(title,isOn:Binding(get:{draft[key] == .bool(true)},set:{draft[key] = .bool($0)})).toggleStyle(.checkbox).accessibilityIdentifier("text-encoding."+key)}
    private func publish(){context.setNumericPending(origin:origin,key:"textEncoding",pending:pending)}
    private func reload(){fieldGeneration=UUID();pendingFields=[];context.setNumericPending(origin:origin,key:"textEncoding",pending:false);draft=node.config;original=node.config;origin=context.numericOrigin(key:"textEncoding");failure=nil;publish()}
    private func apply(){do {guard let definition=context.operatorDefinition else {throw DataFlowContractError("Text encoding is unavailable.")};try TextEncodingConfiguration.validate(draft,definition:definition);if context.applyTableConfiguration(draft,expected:original) {original=draft;failure=nil;publish()} else {failure=context.statusMessage}} catch {failure=error.localizedDescription}}
    private func loadVocabulary(){
        let panel=NSOpenPanel();panel.allowedContentTypes=[.json];panel.allowsMultipleSelection=false;panel.canChooseDirectories=false
        guard panel.runModal() == .OK,let url=panel.url else {return}
        let scoped=url.startAccessingSecurityScopedResource();defer {if scoped {url.stopAccessingSecurityScopedResource()}}
        do {
            let size=try url.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? Int.max
            guard size<=262144 else {throw DataFlowContractError("Vocabulary JSON must be at most 256 KiB.")}
            let bytes=try Data(contentsOf:url,options:.mappedIfSafe)
            guard bytes.count<=262144 else {throw DataFlowContractError("Vocabulary file changed or exceeds its bound.")}
            let tokens=try JSONDecoder().decode([String].self,from:bytes)
            guard !tokens.isEmpty,tokens.count<=65536,Set(tokens).count==tokens.count,tokens.allSatisfy({!$0.isEmpty}) else {throw DataFlowContractError("Load a JSON array of distinct, nonempty tokens in ID order.")}
            draft["vocabulary"] = .array(tokens.map(JSONValue.string));failure=nil
        }catch {failure=error.localizedDescription}
    }
}

struct TextEncodingInformationView:View {
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Document features and token IDs").font(.headline)
            Text("Document features fits TF-IDF vocabulary/IDF on Training only. Output keeps original text, stable observation identity and nonzero-term counts; the Estimator builds sparse features in the isolated worker. Vocabulary lists terms and IDF. Save model preserves the entire processor. Unseen text must reuse that state rather than refitting.")
            Text("Text → token IDs").font(.headline)
            Text("Character uses Unicode code points. Word + whitespace retains whitespace as tokens; normalization, case and whitespace policies run first. Output is a typed integer sequence and its length. Empty text stays an empty sequence unless beginning/end tokens are enabled.")
            Text("Vocabulary ownership").font(.subheadline.weight(.semibold))
            Text("Fit on Training accepts Training plus optional Validation/Test tables from the same Partition. Held-out text never changes the vocabulary. Supplied vocabulary accepts one Table and a local JSON array in ID order; reserved special tokens precede supplied tokens. Vocabulary limit includes special tokens. Unknown IDs use <unk> only when explicitly enabled.")
            Text("Bounded and explicit").font(.subheadline.weight(.semibold))
            Text("Each observation is limited to 1 MiB text and 65,536 tokens. Read smaller records before encoding larger documents. No text is silently truncated. This tool does not create windows, shift targets, pad sequences or train an embedding. Train maps numeric or token columns into declared tensor shapes with explicit length policies.")
            Text("Apply saves one undoable configuration change. Run Flow computes the output and vocabulary tables. Source and vocabulary versions remain part of captured preparation; optional run-detail capture does not change model requirements.")
        }.font(.caption).textSelection(.enabled)
    }
}
