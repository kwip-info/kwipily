import Foundation

/// A new editable architecture. Existing monolithic Transformer sources and
/// checkpoints keep their original constructor/state identity; no implicit migration.
public enum TransformerBlock {
    public static let reference=CompositeDefinitionReference("kwip.transformer.container")
    public static func definition(values:[String:JSONValue]=[:])throws->CompositeDefinition {
        let full=try AttentionConfiguration(componentID:"transformer",values:values).values
        let shared=["d_model","nhead","dim_feedforward","dropout","activation","layer_norm_eps","norm_first","bias"]
        var encoder=Dictionary(uniqueKeysWithValues:shared.compactMap {key in full[key].map {(key,$0)}})
        var decoder=encoder
        encoder["num_layers"]=full["num_encoder_layers"];decoder["num_layers"]=full["num_decoder_layers"]
        encoder["final_norm"]=true;decoder["final_norm"]=true
        encoder["mask_mode"]=full["source_mask_mode"];encoder["padding_mask_mode"]=full["source_padding_mode"]
        for key in ["target_mask_mode","memory_mask_mode","target_padding_mode","memory_padding_mode"] {decoder[key]=full[key]}
        encoder=try AttentionConfiguration(componentID:"transformer_encoder",values:encoder).values
        decoder=try AttentionConfiguration(componentID:"transformer_decoder",values:decoder).values
        var inputs:[CompositeInput]=[.init("source",targets:[.init("encoder","in")]),.init("target",targets:[.init("decoder","target")])]
        let enc=try AttentionConfiguration(componentID:"transformer_encoder",values:encoder)
        let dec=try AttentionConfiguration(componentID:"transformer_decoder",values:decoder)
        for port in enc.configuredInputs where port != "in" {inputs.append(.init(port=="mask" ? "src_mask":port,targets:[.init("encoder",port)]))}
        for port in dec.configuredInputs where !["target","memory"].contains(port) {inputs.append(.init(port,targets:[.init("decoder",port)]))}
        return .init(reference:reference,title:"Transformer",body:.init(nodes:[
            .init(id:"encoder",type:"transformer_encoder",name:"Encode source context",x:80,y:160,config:encoder),
            .init(id:"decoder",type:"transformer_decoder",name:"Decode target with memory",x:430,y:160,config:decoder)
        ],edges:[.init(id:"encoded-memory",source:"encoder",target:"decoder",targetPort:"memory")]),inputs:inputs,outputs:[.init("out",source:.init("decoder","out"))],composition:.init(kind:.graph,children:[.init(localID:"encoder"),.init(localID:"decoder")]),appearance:.transformer)
    }
}

extension WorkflowExamples {
    public static func preparedEditableTransformerClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        var result=try preparedTransformerClassifier(projectID:projectID,device:device)
        guard let index=result.nodes.firstIndex(where:{if case .modelPrimitive("transformer",_,_)=$0.payload {return true};return false}),
              case .modelPrimitive(_,_,let values)=result.nodes[index].payload else {throw GraphConfigurationError("The Transformer practice source is missing.")}
        let block=try TransformerBlock.definition(values:values)
        result.schemaVersion=max(result.schemaVersion,11)
        result.definitions.append(block);result.nodes[index].payload = .modelBlock(reference:block.reference,overrides:[:])
        let x=result.nodes[index].x
        for i in result.nodes.indices where result.nodes[i].address.domain == .model && result.nodes[i].x>x {result.nodes[i].x+=1000}
        result.name="Prepare and train editable Transformer classifier"
        return result
    }
}
