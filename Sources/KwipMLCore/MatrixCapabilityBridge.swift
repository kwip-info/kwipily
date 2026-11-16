import Foundation

extension BuiltInCapabilityBridge {
    static func matrixDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        let reviewed=MatrixMultiplyComponent.definition
        guard definition == reviewed else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Matrix multiplication differs from its reviewed two-operand contract.")
        }
        func port(_ id:String,_ direction:PortDirection)->PortContract {
            PortContract(id,direction:direction,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:3,maximum:8))),phases:[.forward],maximumConnections:direction == .input ? 1:nil)
        }
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(),packageID:packageID,backend:.pytorch,
            ports:definition.inputs.map {port($0,.input)}+definition.outputs.map {port($0,.output)},fields:[],
            lifecycle:LifecycleContract(phases:[.initialize,.forward,.export],state:[]),devices:[.cpu,.mps],
            availability:.existingPOCAdapter,adapterID:"kwip.runner.graph-v2.matrixMultiply",
            learning:LearningContract(conceptIDs:["linear-algebra.matrix-contraction","linear-algebra.broadcast-gradient-reduction"],lessonIDs:MatrixLearningTask.allCases.map { "assessment." + $0.rawValue },exampleIDs:["prepared-matrix-classifier"]),shapeRuleID:"kwip.shape.matmul.v1")
    }
}
