import Foundation

public enum AttentionLearningTask: String, CaseIterable, Codable {
    case multihead = "multihead-attention-connections"
    case transformer = "transformer-source-target"
    case encoder = "transformer-encoder-context"
    case decoder = "transformer-decoder-memory"
    case scaled = "scaled-attention-similarity"
    public var componentID: String { switch self { case .multihead: return "multihead_attention"; case .transformer: return "transformer"; case .encoder: return "transformer_encoder"; case .decoder: return "transformer_decoder"; case .scaled: return "scaled_dot_product_attention" } }
    public var componentTitle: String { switch self { case .multihead: return "MultiheadAttention"; case .transformer: return "Transformer"; case .encoder: return "TransformerEncoder"; case .decoder: return "TransformerDecoder"; case .scaled: return "Scaled attention" } }
    public var title: String { componentTitle + " · " + (self == .multihead ? "Choose which values matter" : self == .transformer ? "Combine a source with a requested sequence" : self == .encoder ? "Compare positions within a sequence" : self == .decoder ? "Use a source to build a requested sequence" : "Mix values using similarity") }
    public var sourceRoles: [String] { switch self { case .multihead, .scaled: return ["query", "key", "value"]; case .transformer: return ["source", "target"]; case .encoder: return ["in"]; case .decoder: return ["target", "memory"] } }
    public var outputRoles: [String] { self == .multihead ? ["result", "weights"] : ["result"] }
    public var maskReferenceRoles: [String] { switch self { case .multihead, .scaled: return ["query", "key"]; case .transformer: return ["target", "source"]; case .encoder: return ["in", "in"]; case .decoder: return ["target", "memory"] } }
    public var maskInputPort: String { self == .transformer || self == .decoder ? "memory_mask" : self == .multihead ? "attn_mask" : "mask" }
    public var sampleShapes: [String: [Int]] { switch self { case .multihead: return ["query": [2,2], "key": [3,2], "value": [3,2]]; case .scaled: return ["query": [1,2,2], "key": [1,3,2], "value": [1,3,2]]; case .transformer: return ["source": [3,2], "target": [2,2]]; case .encoder: return ["in": [3,2]]; case .decoder: return ["target": [2,2], "memory": [3,2]] } }
    public var outputSampleShapes: [String: [Int]] { self == .multihead ? ["result": [2,2], "weights": [1,2,3]] : ["result": self == .scaled ? [1,2,2] : self == .encoder ? [3,2] : [2,2]] }
    public var checkpointSampleShapes:[String:[Int]] {
        let q=sampleShapes[maskReferenceRoles[0]]!.suffix(2).first!,s=sampleShapes[maskReferenceRoles[1]]!.suffix(2).first!
        return ["blockedConnections":[q,s],"blockedSourcePositions":[s]]
    }
    public func expectedCheckpoints(observations:Int)->[String:LearningTensor] {
        let shape=checkpointSampleShapes["blockedConnections"]!,q=shape[0],s=shape[1]
        return ["blockedConnections":.init(shape:[observations,q,s],values:(0..<observations).flatMap {_ in (0..<q).flatMap {i in (0..<s).map {j in i == 0 && j == 1 ? 1.0:0.0}}}),"blockedSourcePositions":.init(shape:[observations,s],values:Array(repeating:0,count:observations*s))]
    }
    public var maskConfiguration: [String: JSONValue] { ["mode": .string("blocked"), "blocked_queries": .array([0]), "blocked_keys": .array([1]), "padding_keys": .array([])] }
    public var configuration: [String: JSONValue] {
        if self == .multihead { return ["embed_dim": 2, "num_heads": 1, "kdim": 2, "vdim": 2, "dropout": .number(0), "bias": true, "add_bias_kv": false, "add_zero_attn": false, "return_weights": true, "average_weights": false, "attention_mask_mode": .string("supplied"), "padding_mask_mode": .string("supplied")] }
        if self == .scaled { return ["dropout": .number(0), "scale": .null, "mask_mode": .string("supplied")] }
        var c: [String: JSONValue] = ["d_model": 2, "nhead": 1, "dim_feedforward": 3, "dropout": .number(0), "activation": .string("gelu"), "layer_norm_eps": .number(1e-5), "norm_first": true, "bias": true]
        if self == .transformer { c.merge(["num_encoder_layers": 1, "num_decoder_layers": 1, "source_mask_mode": .string("none"), "target_mask_mode": .string("causal"), "memory_mask_mode": .string("supplied"), "source_padding_mode": .string("none"), "target_padding_mode": .string("none"), "memory_padding_mode": .string("none")]) { _, b in b } }
        else if self == .encoder { c.merge(["num_layers": 1, "final_norm": true, "mask_mode": .string("supplied"), "padding_mask_mode": .string("none")]) { _, b in b } }
        else { c.merge(["num_layers": 1, "final_norm": true, "target_mask_mode": .string("causal"), "memory_mask_mode": .string("supplied"), "target_padding_mode": .string("none"), "memory_padding_mode": .string("none")]) { _, b in b } }
        return c
    }
    /// Independently declared native ownership for this finite width2/one-layer task.
    public var parameterShapes: [String: [Int]] {
        func attention(_ prefix: String) -> [String: [Int]] { [prefix+"in_proj_weight": [6,2], prefix+"in_proj_bias": [6], prefix+"out_proj.weight": [2,2], prefix+"out_proj.bias": [2]] }
        func layer(_ prefix: String, decoder: Bool = false) -> [String: [Int]] {
            var p = attention(prefix+"self_attn.")
            if decoder { p.merge(attention(prefix+"multihead_attn.")) { _, b in b } }
            p.merge([prefix+"linear1.weight": [3,2], prefix+"linear1.bias": [3], prefix+"linear2.weight": [2,3], prefix+"linear2.bias": [2]]) { _, b in b }
            for i in 1...(decoder ? 3 : 2) { p[prefix+"norm\(i).weight"] = [2]; p[prefix+"norm\(i).bias"] = [2] }
            return p
        }
        if self == .multihead { return attention("") }
        if self == .scaled { return [:] }
        if self == .transformer { var p = layer("encoder.layers.0."); p.merge(layer("decoder.layers.0.", decoder: true)) { _, b in b }; p.merge(["encoder.norm.weight": [2], "encoder.norm.bias": [2], "decoder.norm.weight": [2], "decoder.norm.bias": [2]]) { _, b in b }; return p }
        var p = layer("layers.0.", decoder: self == .decoder); p["norm.weight"] = [2]; p["norm.bias"] = [2]; return p
    }
    public var controlInstructions: String {
        let controls: String
        switch self {
        case .multihead: controls = "Set Model width, Key width and Value width to 2; Heads to 1; Drop probability to 0. Enable Use bias and Output attention weights. Disable Add learned key/value position, Add zero position and Average across heads. Choose Use mask connection for Attention mask and Padding mask. "
        case .scaled: controls = "Set Drop probability to 0 and use Automatic scale (leave Custom scale unset in the assessed lab). Choose Use mask connection for Attention mask. "
        default: controls = "Set Model width to 2, Heads to 1, Feedforward width to 3 and Drop probability to 0. Choose GELU for Activation; set Layer norm epsilon to 0.00001. Enable Normalize before attention and Use bias. " + (self == .transformer ? "Set Encoder layers and Decoder layers to 1. " : "Set Layers to 1 and enable Final normalization. ") + (self == .encoder ? "Choose Use mask connection for Attention mask. " : "Choose Earlier positions only for Target attention and Use mask connection for Memory attention. ")
        }
        return controls + "On Attention mask, choose Blocked pairs for Mask. Add the pair Query 0 / Key 1 and leave Padding positions empty. In the assessed lab, enter the matching Blocked query positions and Blocked key positions lists."
    }
    public var explanation: String { "A sequence is values arranged in order. A query is the position asking for useful information; keys are the positions it compares with; values are the information it mixes. A head is one comparison group. A mask blocks a connection: 1 means blocked and 0 means allowed. Position numbers start at 0. A causal mask only allows the current and earlier positions. " + (self == .scaled ? "This block owns no learned tables; it compares the supplied arrays directly. " : "Learned tables adjust the comparisons and the resulting mixture. ") + (self == .multihead ? "Weights show the share assigned to each allowed key. " : self == .scaled ? "Automatic scale keeps similarity scores balanced for the width. " : "Normalization balances a row's values. Feedforward transforms each position using learned tables; residual addition keeps earlier information in the result. gelu gently reduces negative values. ") + "An influence (gradient) measures how a small change affects the result. The checker measures it; you do not write code or derivatives." }
    public var instructions: String { "Build: add the supplied " + sourceRoles.joined(separator: ", ") + " Input roles, one " + componentTitle + " and one Attention mask. Connect each source to its matching attention port. Connect " + maskReferenceRoles[0] + " to the mask's query and " + maskReferenceRoles[1] + " to its key. Connect mask to " + maskInputPort + (self == .multihead ? " and padding to key_padding_mask" : "") + ". Add Model Output for result from out" + (self == .multihead ? " and a separate weights Model Output from weights" : "") + ". Configure: " + controlInstructions + " Check: choose Check construction. Compare each observation and position in the result" + (self == .multihead ? " and weights tables" : " table") + ". The checker supplies four changed input and learned-table fixtures. " + explanation }
}
