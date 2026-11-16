import Foundation

public enum RemainingActivationCapabilityBridge {
    static func preluDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func logSoftmaxDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func dropoutDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    private static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard RemainingActivationConfiguration.componentIDs.contains(definition.id),
              let entry=BundledActivationComponents.entries[definition.id],entry.published,entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Activation differs from its reviewed learned-state, axis or parent-mode contract.")
        }
        let ports=[PortContract("in",direction:.input,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:2,maximum:8))),phases:[.forward],maximumConnections:1),
                   PortContract("out",direction:.output,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:2,maximum:8))),phases:[.forward],maximumConnections:nil)]
        let state:[StateDeclaration]=definition.id == "prelu" ? [.init("parameters",kind:.parameters)] : definition.id == "dropout" ? [.init("runRandomGenerator",kind:.randomGenerator,sharing:.explicitShared)] : []
        let phases:[ContractPhase]=definition.id == "log_softmax" ? [.initialize,.forward,.export] : [.initialize,.forward,.backward,.train,.evaluate,.checkpoint,.resume,.export]
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,2,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,ports:ports,fields:definition.fields,
            lifecycle:LifecycleContract(phases:phases,fitBoundary:definition.id == "log_softmax" ? .notApplicable:.trainingPartitionOnly,state:state,resume:definition.id == "log_softmax" ? .none:.weightsOnly),devices:[.cpu,.mps],availability:.existingPOCAdapter,
            adapterID:"kwip.runner.graph-v2.\(definition.id)",learning:LearningContract(conceptIDs:["neural-networks.activation-behavior","neural-networks.training-and-evaluation"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.shape.\(definition.id).v1")
    }
}
