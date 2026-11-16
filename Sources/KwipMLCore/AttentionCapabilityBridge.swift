import Foundation

/// Declared ceilings are specialized by checked observation-first attention geometry.
public enum AttentionCapabilityBridge {
    static func multiheadAttentionDeclaration(_ d:ComponentDefinition)throws->CapabilityDescriptor {try declaration(d)}
    static func transformerDeclaration(_ d:ComponentDefinition)throws->CapabilityDescriptor {try declaration(d)}
    static func transformerEncoderDeclaration(_ d:ComponentDefinition)throws->CapabilityDescriptor {try declaration(d)}
    static func transformerDecoderDeclaration(_ d:ComponentDefinition)throws->CapabilityDescriptor {try declaration(d)}
    static func scaledDotProductAttentionDeclaration(_ d:ComponentDefinition)throws->CapabilityDescriptor {try declaration(d)}
    static func attentionMaskDeclaration(_ d:ComponentDefinition)throws->CapabilityDescriptor {try declaration(d)}
    static func rotaryPositionDeclaration(_ d:ComponentDefinition)throws->CapabilityDescriptor {try declaration(d)}
    private static func declaration(_ d:ComponentDefinition)throws->CapabilityDescriptor {
        guard let entry=BundledAttentionComponents.entries[d.id],entry.published,entry.definition == d else {throw BuiltInCapabilityBridgeError.incompatibleRegistry("Attention differs from its reviewed masks, widths or named output contract.")}
        if d.id=="rotary_position" {
            let inputs=d.inputs.map {id in PortContract(id,direction:.input,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:3,maximum:4))),phases:[.forward],required:true,maximumConnections:1)}
            let outputs=d.outputs.map {PortContract($0,direction:.output,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:3,maximum:4))),phases:[.forward])}
            return .init(id:d.id,version:ContractVersion(),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,ports:inputs+outputs,fields:d.fields,lifecycle:.init(phases:[.initialize,.forward,.backward,.export]),devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.graph-v2.rotary_position",learning:.init(conceptIDs:["attention.relative-position","linear-algebra.rotation"],lessonIDs:["rotary_position"],exampleIDs:["prepared-rotary-position-classifier"]),shapeRuleID:"kwip.shape.rotary_position.v1")
        }
        let utility=d.id == "attention_mask", parameterless=utility || d.id == "scaled_dot_product_attention"
        let masks=Set(entry.inputConditions.keys)
        func port(_ id:String,_ direction:PortDirection)->PortContract {
            let integer=utility && direction == .output || masks.contains(id)
            let dimensions:[DimensionContract]
            if integer {dimensions=id.contains("padding") ? [.symbol("sourcePositions")]:[.symbol("queryPositions"),.symbol("sourcePositions")]}
            else if id == "weights" {dimensions=[.symbol("queryPositions"),.symbol("sourcePositionsWithAugmentation")]}
            else {dimensions=[.symbol("positions"),.symbol("values")]}
            return PortContract(id,direction:direction,schema:.tensor(TensorContract(dimensions,dtype:integer ? .int64:.float32,leadingAxes:.init(minimum:1,maximum:(d.id == "scaled_dot_product_attention" || utility || id == "weights") ? 2:1))),phases:[.forward],required:direction == .output || !masks.contains(id),maximumConnections:direction == .input ? 1:nil)
        }
        var state:[StateDeclaration]=parameterless ? []:[StateDeclaration("parameters",kind:.parameters)]
        if !utility {state.append(StateDeclaration("runRandomGenerator",kind:.randomGenerator,sharing:.explicitShared))}
        let phases:[ContractPhase]=utility ? [.initialize,.forward,.export]:[.initialize,.forward,.backward,.train,.evaluate,.checkpoint,.resume,.export]
        return CapabilityDescriptor(id:d.id,version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,ports:d.inputs.map {port($0,.input)}+d.outputs.map {port($0,.output)},fields:d.fields,lifecycle:LifecycleContract(phases:phases,fitBoundary:utility ? .notApplicable:.trainingPartitionOnly,state:state,resume:utility ? .none:.weightsOnly),devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.graph-v2.\(d.id)",learning:LearningContract(conceptIDs:["attention.comparison-groups","attention.blocked-connections"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.shape.\(d.id).v1")
    }
}
