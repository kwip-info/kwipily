import Foundation

extension MixtureExpertsConfiguration {
    public static let runtimeDescriptor:[String:JSONValue]=[
        "version":.object(["major":1,"minor":0,"patch":0]),"adapterID":.string("kwip.runner.graph-v2.mixture_experts"),
        "graphSchemaVersions":.array([2]),"expertMinimum":2,"expertMaximum":4,"maximumRank":3,
        "dropless":true,"graphDTypes":.array([.string("float32")]),"maskDTypes":.array([.string("int64")])]
    public static func verifyRuntime(_ data:Data)->Bool {(try? JSONDecoder().decode([String:JSONValue].self,from:data))==runtimeDescriptor}
}

extension BuiltInCapabilityBridge {
    static func mixtureDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard definition.inputs==["in","mask"],definition.outputs==["out"] else {throw BuiltInCapabilityBridgeError.incompatibleRegistry("MoE needs its reviewed tensor and mask interface.")}
        let defaults=Dictionary(uniqueKeysWithValues:definition.fields.map{($0.key,$0.defaultValue)})
        _=try MixtureExpertsConfiguration(values:defaults,registry:.init(components:[definition]+(try ComponentRegistry.builtIn()).components.filter{$0.id != definition.id}))
        func port(_ id:String,_ direction:PortDirection,_ width:String)->PortContract {
            PortContract(id,direction:direction,schema:.tensor(TensorContract([.parameter(width)],dtype:.float32,leadingAxes:.init(minimum:1,maximum:2))),phases:[.forward],maximumConnections:direction == .input ? 1:nil)
        }
        return .init(id:definition.id,version:ContractVersion(),packageID:packageID,backend:.pytorch,
            ports:[port("in",.input,"input_features"),PortContract("mask",direction:.input,schema:.tensor(TensorContract([.symbol("unit")],dtype:.int64,leadingAxes:.init(minimum:1,maximum:1))),phases:[.forward],maximumConnections:1),port("out",.output,"output_features")],fields:definition.fields,
            lifecycle:.init(phases:[.initialize,.forward,.export],state:[.init("parameters",kind:.parameters)]),devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.graph-v2.mixture_experts",
            learning:.init(conceptIDs:["neural-networks.conditional-experts"],lessonIDs:[MixtureExpertsPractice.lessonID],exampleIDs:[MixtureExpertsPractice.lessonID]),shapeRuleID:"kwip.shape.mixture_experts.v1")
    }
}
