import Foundation
import KwipMLCore

/// Display-only grouping; original parameter identities remain the edit authority.
enum AttentionParameterPresentation {
    static func isAttention(_ id:String)->Bool {
        id.contains("layers.") || id.hasPrefix("encoder.") || id.hasPrefix("decoder.") || id.hasPrefix("norm.") || ["in_proj_weight","in_proj_bias","q_proj_weight","k_proj_weight","v_proj_weight","bias_k","bias_v","out_proj.weight","out_proj.bias"].contains(id)
    }
    static func group(_ id:String)->String {
        let parts=id.split(separator:".").map(String.init)
        if let index=parts.firstIndex(of:"layers"),parts.count>index+1,let n=Int(parts[index+1]) {
            let prefix=index>0 ? parts[0].capitalized+" layer":"Layer"
            return prefix+" \(n+1)"
        }
        if id.hasPrefix("encoder.norm.") {return "Encoder final normalization"}
        if id.hasPrefix("decoder.norm.") {return "Decoder final normalization"}
        if id.hasPrefix("norm.") {return "Final normalization"}
        return "Attention"
    }
    static func localTitle(_ id:String)->String {
        var key=id
        let parts=id.split(separator:".").map(String.init)
        if let index=parts.firstIndex(of:"layers"),parts.count>index+2 {key=parts[(index+2)...].joined(separator:".")}
        else if id.hasPrefix("encoder.norm.") || id.hasPrefix("decoder.norm.") {key="norm."+parts.last!}
        var prefix=""
        if key.hasPrefix("self_attn.") {prefix="Self-attention · ";key=String(key.dropFirst(10))}
        else if key.hasPrefix("multihead_attn.") {prefix="Cross-attention · ";key=String(key.dropFirst(15))}
        let names=["in_proj_weight":"Query/key/value projection weights","in_proj_bias":"Query/key/value projection biases","q_proj_weight":"Query projection weights","k_proj_weight":"Key projection weights","v_proj_weight":"Value projection weights","bias_k":"Learned key position","bias_v":"Learned value position","out_proj.weight":"Output projection weights","out_proj.bias":"Output projection biases","linear1.weight":"Feedforward input weights","linear1.bias":"Feedforward input biases","linear2.weight":"Feedforward output weights","linear2.bias":"Feedforward output biases"]
        if let name=names[key] {return prefix+name}
        if key.hasPrefix("norm"),let end=key.split(separator:".").last {
            let scope=key.split(separator:".").first.map(String.init) ?? "norm"
            let n=scope.dropFirst(4)
            return (n.isEmpty ? "":"Normalization \(n) · ")+(end=="weight" ? "Weights":"Biases")
        }
        return key.replacingOccurrences(of:"_",with:" ").capitalized
    }
    static func title(_ id:String)->String {
        let group=group(id),name=localTitle(id)
        return group=="Attention" ? name:group+" · "+name
    }
}
