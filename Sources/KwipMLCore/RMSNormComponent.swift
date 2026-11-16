import Foundation

/// Scale-only RMS normalization shares strict trailing-sample geometry with LayerNorm.
public struct RMSNormConfiguration:Equatable,Sendable {
    private let geometry:LayerNormConfiguration
    public var normalizedShape:[Int] {geometry.normalizedShape}
    public var epsilon:Double {geometry.epsilon}
    public var elementwiseAffine:Bool {geometry.elementwiseAffine}
    public var parameterCount:Int {geometry.parameterCount}
    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==["normalized_shape","eps","elementwise_affine"] else {throw GraphConfigurationError("RMS normalization accepts trailing shape, epsilon and learned scale only; it has no bias or running statistics.")}
        geometry=try LayerNormConfiguration(values:values.merging(["bias":false]){_,new in new})
    }
    public var values:[String:JSONValue] {geometry.values.filter {$0.key != "bias"}}
    public func outputLayout(for input:BatchedTensorLayout)throws->BatchedTensorLayout {try geometry.outputLayout(for:input)}
}

extension BuiltInCapabilityBridge {
    static func rmsNormDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard let entry=BundledNormalizationComponents.entries["rms_norm"],entry.published,definition==entry.definition else {throw BuiltInCapabilityBridgeError.incompatibleRegistry("RMS normalization differs from its reviewed scale-only trailing-shape contract.")}
        let ports=[PortContract("in",direction:.input,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:2,maximum:8))),phases:[.forward],maximumConnections:1),PortContract("out",direction:.output,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:2,maximum:8))),phases:[.forward])]
        return .init(id:definition.id,version:ContractVersion(),packageID:packageID,backend:.pytorch,ports:ports,fields:definition.fields,lifecycle:.init(phases:[.initialize,.forward,.backward,.export],state:[StateDeclaration("parameters",kind:.parameters)]),devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.graph-v2.rms_norm",learning:.init(conceptIDs:["linear-algebra.root-mean-square","neural-networks.learned-scale"],lessonIDs:["rms_norm"],exampleIDs:["prepared-rms-norm-classifier"]),shapeRuleID:"kwip.shape.rms_norm.v1")
    }
}
extension WorkflowExamples {
    public static func preparedRMSNormClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        var graph=ExampleFlows.layerNormClassifier()
        guard let index=graph.nodes.firstIndex(where:{$0.type=="layer_norm"}) else {throw GraphConfigurationError("Missing normalization practice.")}
        graph.nodes[index].type="rms_norm";graph.nodes[index].name="Scale by root mean square";graph.nodes[index].config.removeValue(forKey:"bias")
        graph.name="RMS normalization · scale without centering"
        return try preparedClassifier(projectID:projectID,device:device,model:graph)
    }
}

/// Exact discovery is separate from the stateful normalization family.
public struct RMSNormRuntimeCapability:Equatable {
    private init() {}
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=4096,let entry=BundledNormalizationComponents.entries["rms_norm"],entry.published,try JSONDecoder().decode(JSONValue.self,from:data)==entry.runtimeContract else {throw GraphConfigurationError("RMS normalization runtime differs from the reviewed scale-only, non-centering contract.")}
        return Self()
    }
}
