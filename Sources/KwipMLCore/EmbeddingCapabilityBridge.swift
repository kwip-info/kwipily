import Foundation

extension BuiltInCapabilityBridge {
    static func embeddingDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        let source=definition.id=="index_input"
        let reviewed=source ? EmbeddingComponent.indexInputDefinition : EmbeddingComponent.definition
        guard definition == reviewed else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("\(definition.id) differs from its reviewed lookup configuration/ports.")
        }
        func port(_ id:String,_ direction:PortDirection)->PortContract {
            let indices=source || direction == .input
            let tensor=TensorContract([],dtype:indices ? .int64 : .float32,
                leadingAxes:.init(minimum:indices ? 2 : 3,maximum:indices ? (source ? 8 : 7) : 8))
            return PortContract(id,direction:direction,schema:.tensor(tensor),phases:[.forward],maximumConnections:direction == .input ? 1 : nil)
        }
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(),packageID:packageID,backend:.pytorch,
            ports:definition.inputs.map {port($0,.input)}+definition.outputs.map {port($0,.output)},
            fields:definition.fields,lifecycle:LifecycleContract(phases:[.initialize,.forward,.export],
                state:source ? [] : [StateDeclaration("parameters",kind:.parameters)]),
            devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.graph-v2.\(definition.id)",
            learning:LearningContract(conceptIDs:source ? ["linear-algebra.integer-indices"] : ["linear-algebra.row-gather","linear-algebra.scatter-add-gradients"],lessonIDs:EmbeddingLearningTask.allCases.map { "assessment." + $0.rawValue },exampleIDs:["prepared-embedding-classifier"]),
            shapeRuleID:"kwip.shape.\(definition.id).v1")
    }
}
