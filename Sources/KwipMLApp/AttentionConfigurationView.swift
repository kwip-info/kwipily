import SwiftUI
import KwipMLCore

struct AttentionMaskRow:Identifiable,Equatable {
    let id=UUID()
    var query:String
    var key:String
}
struct AttentionInspectorDraft:Equatable {
    let componentID:String
    var values:[String:JSONValue]
    var numeric:[String:String]=[:]
    var optionalEnabled:[String:Bool]=[:]
    var blockedPairs:[AttentionMaskRow]=[]
    var paddingPositions:[AttentionMaskRow]=[]
    init(componentID:String,values:[String:JSONValue])throws {
        self.componentID=componentID;self.values=try AttentionConfiguration(componentID:componentID,values:values).values
        for key in ["kdim","vdim","scale"] where self.values[key] != nil {optionalEnabled[key]=self.values[key] != .null}
        func positions(_ key:String)->[String] {if case .array(let items)=self.values[key] {return items.map {String($0.integerValue!)}};return []}
        blockedPairs=zip(positions("blocked_queries"),positions("blocked_keys")).map {.init(query:$0.0,key:$0.1)}
        paddingPositions=positions("padding_keys").map {.init(query:"",key:$0)}
    }
    func text(_ key:String)->String {
        if let text=numeric[key] {return text}
        if let n=values[key]?.integerValue {return String(n)}
        if let n=values[key]?.doubleValue {return String(n)}
        if ["kdim","vdim"].contains(key) {return String(values["embed_dim"]?.integerValue ?? 4)}
        return key=="scale" ? "1":""
    }
    mutating func setText(_ text:String,key:String) {numeric[key]=text}
    func configuration()throws->[String:JSONValue] {
        var result=values
        let integerKeys:Set<String>=["embed_dim","num_heads","kdim","vdim","d_model","nhead","dim_feedforward","num_layers","num_encoder_layers","num_decoder_layers","sequence_axis","rotary_dim","start_position"]
        for key in integerKeys.union(["dropout","scale","layer_norm_eps","base","position_scale"]) where values[key] != nil {
            if optionalEnabled[key]==false {result[key] = .null;continue}
            let text=text(key).trimmingCharacters(in:.whitespacesAndNewlines)
            if integerKeys.contains(key) {
                guard let n=Int64(text) else {throw GraphConfigurationError("Enter a whole number for widths, heads and layer counts.")};result[key] = .integer(n)
            } else {
                guard let n=Double(text),n.isFinite else {throw GraphConfigurationError("Enter a finite numeric value.")};result[key] = .number(n)
            }
        }
        if componentID=="attention_mask" {
            func list(_ values:[String])throws->JSONValue {.array(try values.map {value in guard let n=Int64(value.trimmingCharacters(in:.whitespacesAndNewlines)) else {throw GraphConfigurationError("Mask positions must be whole numbers.")};return .integer(n)})}
            result["blocked_queries"]=try list(blockedPairs.map(\.query));result["blocked_keys"]=try list(blockedPairs.map(\.key));result["padding_keys"]=try list(paddingPositions.map(\.key))
        }
        return try AttentionConfiguration(componentID:componentID,values:result).values
    }
}

struct AttentionConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var inputLayouts:[String:BatchedTensorLayout]=[:]
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:([String:JSONValue],[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=try! AttentionInspectorDraft(componentID:"multihead_attention",values:[:])
    @State private var baseline:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var rejection:String?
    @State private var loadFailed=false
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.layers.primary:ComponentVisualFamily.layers.secondary}
    private var candidate:[String:JSONValue]? {guard !loadFailed,draft.componentID==node.type else {return nil};return try? draft.configuration()}
    private var changed:Bool {candidate.map {$0 != baseline} ?? true}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            if loadFailed {
                Label(rejection ?? "Restore valid source settings before editing this component.",systemImage:"exclamationmark.circle").foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)
                Button("Reload settings") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy)
            } else {
            if node.type=="rotary_position" {rotary}
            if node.type=="multihead_attention" {multihead}
            else if node.type=="scaled_dot_product_attention" {scaled}
            else if node.type=="attention_mask" {maskEditor}
            if node.type=="transformer_encoder" {encoder}
            if node.type=="transformer_decoder" {decoder}
            if node.type=="transformer" {transformer}
            actions
            }
        }.font(.caption).tint(accent).toggleStyle(.checkbox)
            .onAppear {reload()}.onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node) {_,next in
                guard next.config != original else {return}
                if !changed {reload()} else {rejection="Saved settings changed. Revert to load them; your draft is preserved."}
            }.onChange(of:changed) {_,value in onPending(value)}.onDisappear {onPending(false)}
    }
    private var multihead:some View {
        VStack(alignment:.leading,spacing:10) {
            number("Model width",key:"embed_dim");number("Heads",key:"num_heads")
            section("Key and value widths",id:"widths") {
                optional("Use model width for keys",key:"kdim",inverted:true)
                if draft.optionalEnabled["kdim"]==true {number("Key width",key:"kdim")}
                optional("Use model width for values",key:"vdim",inverted:true)
                if draft.optionalEnabled["vdim"]==true {number("Value width",key:"vdim")}
            }
            number("Drop probability",key:"dropout");flag("Use bias",key:"bias")
            section("Masks and weights",id:"masks",expanded:true) {
                mask("Attention mask",key:"attention_mask_mode",causal:true)
                mask("Padding mask",key:"padding_mask_mode")
                flag("Output attention weights",key:"return_weights")
                flag("Average across heads",key:"average_weights").disabled(draft.values["return_weights"] != .bool(true))
            }
            section("Source additions",id:"additions") {
                flag("Add learned key/value position",key:"add_bias_kv")
                flag("Add zero position",key:"add_zero_attn")
            }
        }
    }
    private var encoder:some View {
        VStack(alignment:.leading,spacing:10) {
            number("Model width",key:"d_model");number("Heads",key:"nhead");number("Layers",key:"num_layers")
            transformerShared
            section("Masks",id:"encoder-masks",expanded:true) {
                mask("Attention mask",key:"mask_mode",causal:true)
                mask("Padding mask",key:"padding_mask_mode")
            }
        }
    }
    private var decoder:some View {
        VStack(alignment:.leading,spacing:10) {
            number("Model width",key:"d_model");number("Heads",key:"nhead");number("Layers",key:"num_layers")
            transformerShared
            section("Masks",id:"decoder-masks",expanded:true) {
                mask("Target attention",key:"target_mask_mode",causal:true)
                mask("Memory attention",key:"memory_mask_mode")
                mask("Target padding",key:"target_padding_mode")
                mask("Memory padding",key:"memory_padding_mode")
            }
        }
    }
    private var transformer:some View {
        VStack(alignment:.leading,spacing:10) {
            number("Model width",key:"d_model");number("Heads",key:"nhead")
            number("Encoder layers",key:"num_encoder_layers");number("Decoder layers",key:"num_decoder_layers")
            transformerShared
            section("Masks",id:"transformer-masks") {
                mask("Source attention",key:"source_mask_mode",causal:true)
                mask("Target attention",key:"target_mask_mode",causal:true)
                mask("Memory attention",key:"memory_mask_mode")
                mask("Source padding",key:"source_padding_mode")
                mask("Target padding",key:"target_padding_mode")
                mask("Memory padding",key:"memory_padding_mode")
            }
        }
    }
    private var transformerShared:some View {
        VStack(alignment:.leading,spacing:10) {
            number("Feedforward width",key:"dim_feedforward");number("Drop probability",key:"dropout")
            choice("Activation",key:"activation",options:[.init(id:"relu",title:"ReLU"),.init(id:"gelu",title:"GELU")])
            flag("Use bias",key:"bias")
            section("Normalization",id:"normalization") {
                number("Layer norm epsilon",key:"layer_norm_eps")
                flag("Normalize before attention",key:"norm_first")
                if node.type != "transformer" {flag("Final normalization",key:"final_norm")}
            }
        }
    }
    private var rotary:some View {
        VStack(alignment:.leading,spacing:10) {
            if !inputLayouts.isEmpty {
                VStack(alignment:.leading,spacing:5) {
                    Text("Inputs · planned").fontWeight(.semibold).foregroundStyle(accent)
                    ForEach(["query","key"],id:\.self) {port in
                        if let layout=inputLayouts[port] {Text(port.capitalized+" [B, "+layout.sample.dimensions.map(String.init).joined(separator:", ")+"]").font(.caption.monospacedDigit()).foregroundStyle(MLTheme.muted)}
                    }
                }
            }
            let rank=inputLayouts["query"].map {$0.sample.dimensions.count+1}
            let axes=rank.map {Array(1..<($0-1))} ?? [1,2]
            VStack(alignment:.leading,spacing:4) {
                Text("Sequence axis")
                FlatChoiceControl(title:"Sequence axis",selection:String(Int(draft.text("sequence_axis")) ?? -2),options:[FlatChoiceOption(id:"-2",title:"Second-last axis · −2")]+((rank==nil || rank==4) ? [.init(id:"-3",title:"Third-last axis · −3")]:[])+axes.map {.init(id:String($0),title:"Axis \($0)")}) {draft.setText($0,key:"sequence_axis");rejection=nil}.disabled(isBusy || boundFields.contains("sequence_axis"))
            }
            choice("Rotation",key:"rotation",options:[.init(id:"full",title:"Full feature width"),.init(id:"custom",title:"Custom width")])
            if draft.values["rotation"] == .string("custom") {number("Rotated width",key:"rotary_dim")}
            section("Rotation options",id:"rotary-options",expanded:true) {
                choice("Pairing",key:"pairing",options:[.init(id:"adjacent",title:"Adjacent pairs · Llama"),.init(id:"halves",title:"Split halves")])
                number("Frequency base",key:"base")
            }
            section("Positions",id:"rotary-positions",expanded:true) {
                number("Starting position",key:"start_position")
                number("Position scale",key:"position_scale")
            }
        }
    }
    private var scaled:some View {
        VStack(alignment:.leading,spacing:10) {
            number("Drop probability",key:"dropout")
            optional("Automatic scale",key:"scale",inverted:true)
            if draft.optionalEnabled["scale"]==true {number("Custom scale",key:"scale")}
            mask("Attention mask",key:"mask_mode",causal:true)
        }
    }
    private var maskEditor:some View {
        VStack(alignment:.leading,spacing:10) {
            choice("Mask",key:"mode",options:[.init(id:"none",title:"Allow all"),.init(id:"causal",title:"Earlier positions only"),.init(id:"blocked",title:"Blocked pairs")])
            if draft.values["mode"] == .string("blocked") || !draft.blockedPairs.isEmpty {
                Text("Blocked query/key pairs").fontWeight(.semibold).foregroundStyle(accent)
                HStack {Text("Query").frame(maxWidth:.infinity,alignment:.leading);Text("Key").frame(maxWidth:.infinity,alignment:.leading);Color.clear.frame(width:22,height:1)}
                ForEach($draft.blockedPairs) {$row in
                    HStack(spacing:6) {
                        TextField("Query",text:$row.query).flatField().accessibilityLabel("Query position")
                        TextField("Key",text:$row.key).flatField().accessibilityLabel("Key position")
                        Button {draft.blockedPairs.removeAll {$0.id==row.id}} label:{Image(systemName:"trash")}.buttonStyle(.plain).accessibilityLabel("Remove blocked pair")
                    }
                }
                Button("Add pair") {draft.blockedPairs.append(.init(query:"0",key:"0"))}.buttonStyle(FlatButtonStyle()).disabled(draft.blockedPairs.count>=4096 || draft.values["mode"] != .string("blocked"))
            }
            Text("Padding positions").fontWeight(.semibold).foregroundStyle(accent)
            ForEach($draft.paddingPositions) {$row in
                HStack(spacing:6) {
                    TextField("Key position",text:$row.key).flatField().accessibilityLabel("Padding key position")
                    Button {draft.paddingPositions.removeAll {$0.id==row.id}} label:{Image(systemName:"trash")}.buttonStyle(.plain).accessibilityLabel("Remove padding position")
                }
            }
            Button("Add position") {draft.paddingPositions.append(.init(query:"",key:"0"))}.buttonStyle(FlatButtonStyle()).disabled(draft.paddingPositions.count>=4096)
        }.disabled(isBusy || !boundFields.isDisjoint(with:["mode","blocked_queries","blocked_keys","padding_keys"]))
    }
    private var actions:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(isBusy || !changed || candidate==nil).accessibilityIdentifier("attention.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("attention.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied attention changes")}
            }
            if let rejection {Text(rejection).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
            else if candidate==nil {Text(validationMessage).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
        }
    }
    private var validationMessage:String {do {_=try draft.configuration();return ""} catch {return error.localizedDescription}}
    private func section<C:View>(_ title:String,id:String,expanded:Bool=false,@ViewBuilder content:@escaping()->C)->some View {
        InspectorSection(title,symbol:"slider.horizontal.3",identifier:"attention."+id,initiallyExpanded:expanded,accent:accent,content:content)
    }
    private func choice(_ title:String,key:String,options:[FlatChoiceOption])->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:draft.values[key]?.stringValue ?? "",options:options,accent:accent) {draft.values[key] = .string($0);rejection=nil}.disabled(isBusy || boundFields.contains(key))}
    }
    private func mask(_ title:String,key:String,causal:Bool=false)->some View {
        choice(title,key:key,options:[.init(id:"none",title:"Allow all")]+(causal ? [.init(id:"causal",title:"Earlier positions only")]:[])+[.init(id:"supplied",title:"Use mask connection")])
    }
    private func number(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);TextField(title,text:Binding(get:{draft.text(key)},set:{draft.setText($0,key:key);rejection=nil})).flatField().frame(maxWidth:.infinity).disabled(isBusy || boundFields.contains(key)).accessibilityLabel(title)}
    }
    private func flag(_ title:String,key:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.values[key]?.boolValue ?? false},set:{draft.values[key] = .bool($0);rejection=nil})).disabled(isBusy || boundFields.contains(key)).accessibilityIdentifier("attention."+key)
    }
    private func optional(_ title:String,key:String,inverted:Bool=false)->some View {
        Toggle(title,isOn:Binding(get:{(draft.optionalEnabled[key] ?? false) != inverted},set:{draft.optionalEnabled[key]=$0 != inverted;rejection=nil})).disabled(isBusy || boundFields.contains(key))
    }
    private func reload() {
        original=node.config;loadFailed=false
        do {
            let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:configurationVersion,registry:registry)
            let next=try AttentionInspectorDraft(componentID:node.type,values:values)
            baseline=try next.configuration();draft=next;rejection=nil;onPending(false)
        } catch {loadFailed=true;rejection=error.localizedDescription;onPending(false)}
    }
    private func apply() {
        guard !isBusy,changed,let values=candidate else {return}
        guard onApply(values,original) else {rejection="Settings were not applied. Check connected or exposed masks/weights, authored parameter policies and selection.";return}
        baseline=values;original=values;if let next=try? AttentionInspectorDraft(componentID:node.type,values:values) {draft=next};rejection=nil;onPending(false)
    }
}
