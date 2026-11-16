import Foundation

/// Checked observation-first attention geometry. Masks are integer connections; runtime owns their contents.
public struct AttentionConfiguration:Equatable,Sendable {
    public static let componentIDs:Set<String>=["multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention","attention_mask","rotary_position"]
    public static let maximumElements=100_000_000
    public static let maximumParameters=2_000_000
    public struct Parameter:Equatable,Sendable {public let name:String;public let shape:[Int]}
    public struct Resolved:Equatable,Sendable {
        public let outputs:[String:BatchedTensorLayout]
        public let outputDTypes:[String:ContractDType]
        public let outputElementsPerObservation:Int
        public let scoreElementsPerObservation:Int
        public let maskElementsPerObservation:Int
        public let workPerObservation:Int
    }
    public let componentID:String
    public let values:[String:JSONValue]
    private struct Mask {let field:String;let port:String;let causal:Bool}
    private static let masks:[String:[Mask]]=[
        "multihead_attention":[.init(field:"attention_mask_mode",port:"attn_mask",causal:true),.init(field:"padding_mask_mode",port:"key_padding_mask",causal:false)],
        "transformer":[.init(field:"source_mask_mode",port:"src_mask",causal:true),.init(field:"target_mask_mode",port:"tgt_mask",causal:true),.init(field:"memory_mask_mode",port:"memory_mask",causal:false),.init(field:"source_padding_mode",port:"src_key_padding_mask",causal:false),.init(field:"target_padding_mode",port:"tgt_key_padding_mask",causal:false),.init(field:"memory_padding_mode",port:"memory_key_padding_mask",causal:false)],
        "transformer_encoder":[.init(field:"mask_mode",port:"mask",causal:true),.init(field:"padding_mask_mode",port:"src_key_padding_mask",causal:false)],
        "transformer_decoder":[.init(field:"target_mask_mode",port:"tgt_mask",causal:true),.init(field:"memory_mask_mode",port:"memory_mask",causal:false),.init(field:"target_padding_mode",port:"tgt_key_padding_mask",causal:false),.init(field:"memory_padding_mode",port:"memory_key_padding_mask",causal:false)],
        "scaled_dot_product_attention":[.init(field:"mask_mode",port:"mask",causal:true)],"attention_mask":[]]
    private static let floats:[String:[String]]=["multihead_attention":["query","key","value"],"transformer":["source","target"],"transformer_encoder":["in"],"transformer_decoder":["target","memory"],"scaled_dot_product_attention":["query","key","value"],"attention_mask":["query","key"]]
    public var configuredInputs:[String] {if componentID=="rotary_position" {return try! RotaryPositionConfiguration(values:values).configuredInputs};return Self.floats[componentID]!+Self.masks[componentID]!.filter {text($0.field)=="supplied"}.map(\.port)}
    public var inputDTypes:[String:ContractDType] {if componentID=="rotary_position" {return Dictionary(uniqueKeysWithValues:configuredInputs.map {($0,ContractDType.float32)})};return Dictionary(uniqueKeysWithValues:configuredInputs.map {($0,Self.floats[componentID]!.contains($0) ? .float32:.int64)})}
    public var outputPorts:[String] {if componentID=="rotary_position" {return ["query","key"]};return componentID=="attention_mask" ? ["mask","padding"]:componentID=="multihead_attention" && flag("return_weights") ? ["out","weights"]:["out"]}
    public var stochastic:Bool {(values["dropout"]?.doubleValue ?? 0)>0}
    private func integer(_ key:String)->Int {values[key]!.integerValue!}
    private func flag(_ key:String)->Bool {values[key]?.boolValue ?? false}
    private func text(_ key:String)->String {values[key]?.stringValue ?? "none"}
    private func list(_ key:String)->[Int] {if case .array(let a)=values[key] {return a.map {$0.integerValue!}};return []}
    public func supports(device:String)->Bool {["cpu","mps"].contains(device)}
    public init(componentID:String,values supplied:[String:JSONValue])throws {
        guard Self.componentIDs.contains(componentID) else {throw GraphConfigurationError("Choose a supported attention component.")}
        if componentID=="rotary_position" {self.componentID=componentID;self.values=try RotaryPositionConfiguration(values:supplied).values;return}
        var v:[String:JSONValue]
        switch componentID {
        case "attention_mask":v=["mode":.string("none"),"blocked_queries":.array([]),"blocked_keys":.array([]),"padding_keys":.array([])]
        case "scaled_dot_product_attention":v=["dropout":0,"scale":.null]
        case "multihead_attention":v=["embed_dim":4,"num_heads":2,"dropout":0,"bias":true,"kdim":.null,"vdim":.null,"add_bias_kv":false,"add_zero_attn":false,"return_weights":true,"average_weights":true]
        default:
            v=["d_model":4,"nhead":2,"dim_feedforward":8,"dropout":0,"bias":true,"activation":.string("relu"),"layer_norm_eps":.number(1e-5),"norm_first":false]
            if componentID=="transformer" {v["num_encoder_layers"]=1;v["num_decoder_layers"]=1}
            else {v["num_layers"]=1;v["final_norm"]=false}
        }
        for m in Self.masks[componentID]! {v[m.field] = .string("none")}
        guard Set(supplied.keys).isSubset(of:Set(v.keys)) else {throw GraphConfigurationError("Use only the declared attention controls; weights and execution state cannot be authored.")}
        v.merge(supplied) {_,new in new}
        func whole(_ key:String,_ low:Int,_ high:Int)throws {
            guard let n=v[key]?.doubleValue,n.isFinite,n.rounded(.towardZero)==n,Double(low)<=n,n<=Double(high) else {throw GraphConfigurationError("\(key) must be a whole number from \(low) to \(high).")}
            v[key] = .integer(Int64(n))
        }
        for key in ["embed_dim","d_model"] where v[key] != nil {try whole(key,1,512)}
        for key in ["num_heads","nhead"] where v[key] != nil {try whole(key,1,32)}
        for key in ["kdim","vdim"] where v[key] != nil && v[key] != .null {try whole(key,1,4096)}
        for key in ["num_layers","num_encoder_layers","num_decoder_layers"] where v[key] != nil {try whole(key,1,8)}
        if v["dim_feedforward"] != nil {try whole("dim_feedforward",1,4096)}
        for key in ["bias","add_bias_kv","add_zero_attn","return_weights","average_weights","norm_first","final_norm"] where v[key] != nil {guard v[key]?.boolValue != nil else {throw GraphConfigurationError("\(key) must be a Boolean switch.")}}
        for key in ["dropout","layer_norm_eps","scale"] where v[key] != nil {
            if key=="scale" && v[key] == .null {continue}
            let range=key=="dropout" ? 0.0...1.0:Double(Float.leastNonzeroMagnitude)...Double(Float.greatestFiniteMagnitude)
            guard let n=v[key]?.doubleValue,n.isFinite,range.contains(n) else {throw GraphConfigurationError("Use a finite \(key) value in its declared range.")};v[key] = .number(n)
        }
        if v["activation"] != nil {guard let a=v["activation"]?.stringValue,["relu","gelu"].contains(a) else {throw GraphConfigurationError("Choose ReLU or GELU for Activation.")}}
        if let e=v["embed_dim"]?.integerValue ?? v["d_model"]?.integerValue,let h=v["num_heads"]?.integerValue ?? v["nhead"]?.integerValue {guard e%h==0 else {throw GraphConfigurationError("Model width must divide evenly between Heads.")}}
        for m in Self.masks[componentID]! {guard let t=v[m.field]?.stringValue,(m.causal ? ["none","causal","supplied"]:["none","supplied"]).contains(t) else {throw GraphConfigurationError("Choose a supported blocked-connection arrangement.")}}
        if componentID=="attention_mask" {
            guard let mode=v["mode"]?.stringValue,["none","causal","blocked"].contains(mode) else {throw GraphConfigurationError("Choose none, causal or blocked connections.")}
            var lists:[String:[Int]]=[:]
            for key in ["blocked_queries","blocked_keys","padding_keys"] {
                guard case .array(let items)=v[key],items.count<=4096 else {throw GraphConfigurationError("Mask coordinate lists support at most 4096 entries.")}
                let ints=try items.map {item->Int in guard let n=item.doubleValue,n.isFinite,n.rounded(.towardZero)==n,(0...4095).contains(n) else {throw GraphConfigurationError("Mask positions must be whole numbers from 0 to 4095.")};return Int(n)}
                lists[key]=ints;v[key] = .array(ints.map { .integer(Int64($0)) })
            }
            let q=lists["blocked_queries"]!,k=lists["blocked_keys"]!,p=lists["padding_keys"]!
            guard q.count==k.count,Set(zip(q,k).map {"\($0.0):\($0.1)"}).count==q.count,Set(p).count==p.count,q.isEmpty || mode=="blocked" else {throw GraphConfigurationError("Match unique blocked query/key pairs; source positions must be unique and blocked pairs need Blocked mode.")}
        }
        self.componentID=componentID;self.values=v
        guard parameterCount<=Self.maximumParameters else {throw GraphConfigurationError("Attention exceeds two million learned settings. Reduce widths or layers.")}
    }
    public var parameterShapes:[Parameter] {
        if componentID=="rotary_position" {return []}
        var out:[Parameter]=[]
        func add(_ name:String,_ shape:[Int]) {out.append(.init(name:name,shape:shape))}
        func mha(_ prefix:String,_ e:Int,_ k:Int,_ v:Int,_ augment:Bool=false) {
            if k==e && v==e {add(prefix+"in_proj_weight",[3*e,e])} else {add(prefix+"q_proj_weight",[e,e]);add(prefix+"k_proj_weight",[e,k]);add(prefix+"v_proj_weight",[e,v])}
            if flag("bias") {add(prefix+"in_proj_bias",[3*e])}
            if augment {add(prefix+"bias_k",[1,1,e]);add(prefix+"bias_v",[1,1,e])}
            add(prefix+"out_proj.weight",[e,e]);if flag("bias") {add(prefix+"out_proj.bias",[e])}
        }
        func norm(_ prefix:String,_ e:Int) {add(prefix+"weight",[e]);if flag("bias") {add(prefix+"bias",[e])}}
        func stack(_ prefix:String,_ count:Int,_ decoder:Bool) {
            let e=integer("d_model"),f=integer("dim_feedforward")
            for i in 0..<count {
                let p=prefix+"layers.\(i).";mha(p+"self_attn.",e,e,e)
                if decoder {mha(p+"multihead_attn.",e,e,e)}
                add(p+"linear1.weight",[f,e]);if flag("bias") {add(p+"linear1.bias",[f])}
                add(p+"linear2.weight",[e,f]);if flag("bias") {add(p+"linear2.bias",[e])}
                for j in 1...(decoder ? 3:2) {norm(p+"norm\(j).",e)}
            }
        }
        if componentID=="multihead_attention" {let e=integer("embed_dim");mha("",e,values["kdim"]?.integerValue ?? e,values["vdim"]?.integerValue ?? e,flag("add_bias_kv"))}
        else if componentID=="transformer" {stack("encoder.",integer("num_encoder_layers"),false);norm("encoder.norm.",integer("d_model"));stack("decoder.",integer("num_decoder_layers"),true);norm("decoder.norm.",integer("d_model"))}
        else if ["transformer_encoder","transformer_decoder"].contains(componentID) {stack("",integer("num_layers"),componentID=="transformer_decoder");if flag("final_norm") {norm("norm.",integer("d_model"))}}
        return out
    }
    public var parameterCount:Int {parameterShapes.reduce(0) {$0+$1.shape.reduce(1,*)}}
    public func resolved(layouts:[BatchedTensorLayout])throws->Resolved {
        if componentID=="rotary_position" {return try RotaryPositionConfiguration(values:values).resolved(layouts:layouts)}
        let ports=configuredInputs
        guard layouts.count==ports.count else {throw GraphConfigurationError("Connect exactly the enabled attention inputs and masks.")}
        let source=Dictionary(uniqueKeysWithValues:zip(ports,layouts.map {$0.sample.dimensions}))
        let floats=Self.floats[componentID]!.map {source[$0]!}
        let ranks=componentID=="attention_mask" ? [2,3]:componentID=="scaled_dot_product_attention" ? [3]:[2]
        guard floats.allSatisfy({ranks.contains($0.count) && $0.allSatisfy {(1...4096).contains($0)}}) else {throw GraphConfigurationError("Connect positive observation-first sequence directions up to 4096.")}
        var outputs:[String:BatchedTensorLayout]=[:],dtypes:[String:ContractDType]=[:],scores=0,work=0,masks=0,maskShapes:[String:[Int]]=[:]
        func output(_ port:String,_ shape:[Int],_ dtype:ContractDType = .float32)throws {outputs[port]=try .init(sample:shape);dtypes[port]=dtype}
        func eligible(_ field:String,_ l:Int,_ s:Int)throws {if text(field)=="causal" && l != s {throw GraphConfigurationError("Earlier-positions masks need equal query/source lengths; use a supplied mask for cross comparisons.")}}
        func attention(_ l:Int,_ s:Int,_ e:Int,_ k:Int?=nil,_ v:Int?=nil,_ extra:Int=0,_ project:Bool=true) {
            let h=values["num_heads"]?.integerValue ?? values["nhead"]?.integerValue ?? (componentID=="scaled_dot_product_attention" ? floats[0][0]:1)
            let d=componentID=="scaled_dot_product_attention" ? e:e/h,dv=componentID=="scaled_dot_product_attention" ? floats[2].last!:d
            scores+=h*l*(s+extra);work+=h*l*(s+extra)*(d+dv)
            if project {work+=l*e*e+s*(k ?? e)*e+s*(v ?? e)*e+l*e*e}
        }
        if componentID=="attention_mask" {
            let q=floats[0],k=floats[1]
            guard q.count==k.count,q.count != 3 || (q[0]==k[0] && q[0]<=32) else {throw GraphConfigurationError("Mask references need matching ranks and comparison groups.")}
            let l=q[q.count-2],s=k[k.count-2]
            try eligible("mode",l,s)
            guard list("blocked_queries").allSatisfy({$0<l}), (list("blocked_keys")+list("padding_keys")).allSatisfy({$0<s}) else {throw GraphConfigurationError("Mask positions must lie inside their connected sequences.")}
            try output("mask",[l,s],.int64);try output("padding",[s],.int64);masks=l*s+s;work=masks
        } else if componentID=="scaled_dot_product_attention" {
            let q=floats[0],k=floats[1],v=floats[2];let h=q[0],l=q[1],d=q[2],s=k[1]
            guard h==k[0],h==v[0],h<=32,d==k[2],s==v[1],d<=512,v[2]<=512 else {throw GraphConfigurationError("Match comparison groups, query/key widths and source lengths within supported widths.")}
            try eligible("mask_mode",l,s);attention(l,s,d,nil,nil,0,false);try output("out",[h,l,v[2]]);maskShapes=["mask":[l,s]]
        } else if componentID=="multihead_attention" {
            let q=floats[0],k=floats[1],v=floats[2];let l=q[0],e=q[1],s=k[0]
            guard e==integer("embed_dim"),k[1]==(values["kdim"]?.integerValue ?? e),v[1]==(values["vdim"]?.integerValue ?? e),s==v[0] else {throw GraphConfigurationError("Connect declared query/key/value widths and matching key/value lengths.")}
            try eligible("attention_mask_mode",l,s);let extra=(flag("add_bias_kv") ? 1:0)+(flag("add_zero_attn") ? 1:0)
            attention(l,s,e,k[1],v[1],extra);try output("out",[l,e])
            if flag("return_weights") {try output("weights",flag("average_weights") ? [l,s+extra]:[integer("num_heads"),l,s+extra])}
            maskShapes=["attn_mask":[l,s],"key_padding_mask":[s]]
        } else {
            let e=integer("d_model"),f=integer("dim_feedforward")
            guard floats.allSatisfy({$0.last==e}) else {throw GraphConfigurationError("Every model sequence needs the declared value count.")}
            let s:Int,t:Int,enc:Int,dec:Int
            if componentID=="transformer_encoder" {s=floats[0][0];t=s;enc=integer("num_layers");dec=0}
            else if componentID=="transformer_decoder" {t=floats[0][0];s=floats[1][0];enc=0;dec=integer("num_layers")}
            else {s=floats[0][0];t=floats[1][0];enc=integer("num_encoder_layers");dec=integer("num_decoder_layers")}
            for _ in 0..<enc {attention(s,s,e);work+=2*s*e*f}
            for _ in 0..<dec {attention(t,t,e);attention(t,s,e);work+=2*t*e*f}
            try output("out",[t,e])
            if componentID=="transformer_encoder" {maskShapes=["mask":[s,s],"src_key_padding_mask":[s]]}
            else {maskShapes=["src_mask":[s,s],"tgt_mask":[t,t],"memory_mask":[t,s],"src_key_padding_mask":[s],"tgt_key_padding_mask":[t],"memory_key_padding_mask":[s]]}
        }
        for m in Self.masks[componentID]! {
            if text(m.field)=="supplied" {guard source[m.port]==maskShapes[m.port] else {throw GraphConfigurationError("Supplied binary masks must match the exact connected query/source directions.")};masks+=maskShapes[m.port]!.reduce(1,*)}
            else if text(m.field)=="causal" {masks+=maskShapes[m.port]!.reduce(1,*)}
        }
        let physical=outputs.values.reduce(0) {$0+$1.sample.elementCount}
        // Bounds above limit every factor to4096, layers to8 and groups to32, keeping arithmetic within64-bit Int.
        guard source.values.allSatisfy({$0.reduce(1,*)<=Self.maximumElements}),[physical,scores,masks,work].allSatisfy({$0<=Self.maximumElements}) else {throw GraphConfigurationError("Attention exceeds its tensor, score, mask or computation budget. Reduce widths or sequence lengths.")}
        return .init(outputs:outputs,outputDTypes:dtypes,outputElementsPerObservation:physical,scoreElementsPerObservation:scores,maskElementsPerObservation:masks,workPerObservation:work)
    }
}
