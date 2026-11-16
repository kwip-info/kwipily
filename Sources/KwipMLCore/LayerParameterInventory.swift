import Foundation

public struct LayerParameterChoice:Identifiable,Equatable {
    public let id:String
    public let shape:[Int]
}

public enum LayerParameterInventory {
    public static func choices(type:String,config:[String:JSONValue],version:Int,registry:ComponentRegistry)throws->[LayerParameterChoice] {
        let node=GraphNode(type:type,config:config)
        let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:version,registry:registry)
        func n(_ key:String)->Int {Int(values[key]?.integerValue ?? 0)}
        func dims(_ key:String)->[Int] {if let n=values[key]?.integerValue {return [Int(n)]};if case .array(let rows)=values[key] {return rows.compactMap {$0.integerValue}};return []}
        var result:[LayerParameterChoice]=[]
        func add(_ name:String,_ shape:[Int]) {result.append(.init(id:name,shape:shape))}
        func weight(_ shape:[Int],bias:[Int]?=nil) {add("weight",shape);if values["bias"] == .bool(true),let bias {add("bias",bias)}}
        switch type {
        case "linear":weight([n("out_features"),n("in_features")],bias:[n("out_features")])
        case "lazy_linear":weight([],bias:[n("out_features")])
        case "bilinear":weight([n("out_features"),n("in1_features"),n("in2_features")],bias:[n("out_features")])
        case "embedding","embedding_bag":add("weight",[n("num_embeddings"),n("embedding_dim")])
        case "prelu":add("weight",[n("num_parameters")])
        case "rms_norm":if values["elementwise_affine"] == .bool(true) {add("weight",dims("normalized_shape"))}
        case "layer_norm":if values["elementwise_affine"] == .bool(true) {weight(dims("normalized_shape"),bias:dims("normalized_shape"))}
        case "batch_norm1d","batch_norm2d","instance_norm1d","instance_norm2d","group_norm":
            let c=try StatefulNormalizationConfiguration(componentID:type,values:values)
            if c.affine {add("weight",[c.channelCount]);if c.bias {add("bias",[c.channelCount])}}
        case "rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell":
            for p in try RecurrentConfiguration(componentID:type,values:values).parameterShapes {add(p.name,p.shape)}
        case "multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention":
            for p in try AttentionConfiguration(componentID:type,values:values).parameterShapes {add(p.name,p.shape)}
        case "depthwise_conv2d":
            let c=try DepthwiseConv2dConfiguration(values:values).convolution
            add("weight",[c.outputChannels,c.inputChannels/c.groups]+c.kernelSize);if c.bias {add("bias",[c.outputChannels])}
        case "conv1d","conv2d","conv3d","conv_transpose1d","conv_transpose2d","conv_transpose3d":
            let input=n("in_channels"),output=n("out_channels"),groups=max(1,n("groups"))
            weight((type.hasPrefix("conv_transpose") ? [input,output/groups]:[output,input/groups])+dims("kernel_size"),bias:[output])
        default:break
        }
        return result
    }
}

