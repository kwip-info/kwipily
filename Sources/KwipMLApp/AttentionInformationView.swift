import SwiftUI
import KwipMLCore

struct AttentionInformationView:View {
    let type:String
    var body:some View {
        if type=="attention_mask" {
            LessonRichText(text:"**Reference sequences**\nQuery and key provide sequence lengths, not mask values. Positions start at zero. One means blocked and zero means allowed. Blocked pairs must be unique; padding positions must also be unique. Removing rows is explicit.\n\n**Two outputs**\nMask is an observation × query-position × key-position table. Padding is observation × key-position. Padding positions block their entire key columns in Mask as well as appearing in Padding. Every configured position must fit the connected sequence; combined masks must leave an eligible source for each query.\n\n**Mode changes**\nEarlier positions only requires equal reference lengths. Remove blocked-pair rows explicitly before choosing another mode. Configured choices apply to each observation; arbitrary per-observation masks can be supplied as separate integer tensors.")
        } else {
            LessonRichText(text:sequences)
            LessonRichText(text:"**Heads and masks**\nA head makes an independent set of comparisons. \(type=="scaled_dot_product_attention" ? "All three inputs must have the same head count. ":"Model width must divide evenly between Heads. ")Use mask connection adds a real integer tensor input: one blocks a connection, zero allows it. Attention masks use query × key positions; padding masks use key positions. Earlier positions only permits the current and earlier positions and requires equal query/key lengths. Memory attention has no causal choice. Each query must retain an eligible original source. Disconnect connected or exposed ports explicitly before removing them.\n\n**Dropout and saved state**\nDrop probability is active during training and zero during evaluation. No hidden sequence state or automatic key/value cache is stored between calls. Changing geometry never silently resizes checkpoint weights. CPU supports exact continuation; MPS supports the current explicit weights-only restart policy.\n\n**Limits and output**\nWidths, layer counts, learned-parameter and attention-score budgets are checked before execution. Captured tensor Results contains recorded values only; uncaptured cells are unavailable.")
            if type=="multihead_attention" {
                LessonRichText(text:"**Projections and attention weights**\nKey and value widths can follow Model width or be explicit. Use bias adds learned offsets. Source additions extend the mixing positions with a learned key/value pair or a zero position. Output attention weights exposes the actual mixing shares; Average across heads combines their head axis. Those output weights are results, separate from the learned projection parameters in Weights and biases.")
            } else if type=="scaled_dot_product_attention" {
                LessonRichText(text:"**Scale**\nAutomatic scale uses the reciprocal square root of the query/key width. Custom scale must be positive and finite in float32. This operation has no learned projection parameters or attention-weights output. Dropout still follows training mode.")
            } else {
                LessonRichText(text:"**Normalization and feedforward**\nFeedforward width controls the inner learned transformation at each position. Activation chooses ReLU or GELU. Layer norm epsilon is a positive safety term. Normalize before attention chooses pre-normalization instead of post-normalization. Standalone Encoder/Decoder optionally add Final normalization; the combined Transformer always includes final encoder and decoder normalization.\n\n**Named learned parameters**\nWeights and biases groups parameters by layer, with separate self-attention, cross-attention, feedforward and normalization identities. Removing a layer or bias with an authored policy requires explicitly clearing that policy first. Initialization applies to fresh training; checkpoint loading restores saved parameters.")
            }
        }
    }
    private var sequences:String {
        switch type {
        case "scaled_dot_product_attention":return "**Head-shaped inputs**\nQuery, key and value are already shaped as observation × heads × positions × width. Query/key widths and all head counts must match; key/value lengths must match. Value width may differ. No batch or head axis is inferred or reinterpreted."
        case "multihead_attention":return "**Sequence inputs**\nQuery, key and value use observation × positions × features. Learned projections compare query and key positions and mix the corresponding values. Output preserves the query length and Model width. Key/value lengths must match; their feature widths may differ."
        case "transformer_encoder":return "**One sequence**\nEncoder compares positions within its input. Each layer combines self-attention, a feedforward transformation, residual additions and normalization. Output preserves observation count, sequence length and Model width."
        case "transformer_decoder":return "**Target and memory**\nTarget is the sequence to build. Memory is the encoded sequence it can consult. Each decoder layer has target self-attention and separate cross-attention to memory. Output follows the target length; memory can have a different length. Both widths equal Model width."
        default:return "**Source and target**\nThe encoder summarizes Source; the decoder builds Target using that memory. Encoder layers and Decoder layers are independent counts. Output follows the target length. Both sequences use observation × positions × Model width."
        }
    }
}
