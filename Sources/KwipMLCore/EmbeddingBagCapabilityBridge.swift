import Foundation

public enum EmbeddingBagCapabilityBridge {
    public static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard ["bag_input","embedding_bag"].contains(definition.id),
              let entry=BundledEmbeddingComponents.entries[definition.id],entry.published,
              definition == entry.definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Bag roles and settings must match the reviewed published declaration.")
        }
        let source=definition.id == "bag_input"
        func port(_ id:String,_ direction:PortDirection)->PortContract {
            let reduced=id == "out",floating=reduced || id == "per_sample_weights"
            // Packed item arrays have no observation axis. Concrete packet roles
            // are checked by GraphValidator, never inferred from this rank range.
            let tensor=TensorContract([],dtype:floating ? .float32:.int64,gradients:floating,
                leadingAxes:.init(minimum:reduced ? 2:1,maximum:id == "offsets" ? 1:2))
            return .init(id,direction:direction,schema:.tensor(tensor),phases:[.forward],
                         required:reduced || id == "indices",maximumConnections:direction == .input ? 1:nil)
        }
        return .init(id:definition.id,packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:definition.inputs.map {port($0,.input)}+definition.outputs.map {port($0,.output)},
            fields:definition.fields,lifecycle:.init(phases:[.initialize,.forward,.export],
                state:source ? []:[.init("parameters",kind:.parameters)]),devices:[.cpu,.mps],
            availability:.existingPOCAdapter,adapterID:"kwip.runner.graph-v2."+definition.id,
            learning:.init(conceptIDs:source ? ["data.whole-observation-groups"]:["linear-algebra.row-lookup","linear-algebra.group-reduction"],
                lessonIDs:EmbeddingBagLearningTask.allCases.map {"assessment."+$0.rawValue},
                exampleIDs:["prepared-embedding-bag-classifier"]),shapeRuleID:"kwip.shape."+definition.id+".v1")
    }
}
