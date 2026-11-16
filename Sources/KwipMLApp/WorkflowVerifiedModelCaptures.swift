import Foundation
import KwipMLCore

/// Bounded live samples, verified before publication and resolved by full origin.
/// These are not persisted History artifacts or proof that source files are unchanged.
struct WorkflowVerifiedModelCaptures {
    private let previews: [String: JSONValue]
    private let project: CompositeProjectDocument?
    private let origins: [String: CompositeNodeOrigin]

    init(event: [String: JSONValue], graph: GraphDocument,
         origins: [String: CompositeNodeOrigin], sampleCapture: TrainingSampleCapturePolicy?,
         project: CompositeProjectDocument? = nil, objectiveContext:WorkflowObjectiveArtifactContext? = nil, capturedRegistry:ComponentRegistry? = nil, bagContext:WorkflowBagArtifactContext? = nil, objectiveEvaluationContext:WorkflowObjectiveEvaluationContext? = nil) throws {
        try WorkflowModelTrainingExecution.verifyPresentation(event:event,graph:graph,sampleCapture:sampleCapture,objectiveContext:objectiveContext,capturedRegistry:capturedRegistry,bagContext:bagContext,objectiveEvaluationContext:objectiveEvaluationContext)
        guard Set(origins.keys) == Set(graph.nodes.map(\.id)),
              case .object(let previews) = event["previews"] else { throw WorkflowFileLeaseError.unsafePath }
        self.previews = previews
        self.origins = origins
        self.project = project
    }

    func ports(instancePath: [String], localNodeID: String) -> [String: JSONValue]? {
        let matches = origins.filter { $0.value.instancePath == instancePath && $0.value.localNodeID == localNodeID }
        guard matches.count == 1, let id = matches.first?.key,
              case .object(let preview) = previews[id] else { return nil }
        guard case .object(let outputs) = preview["outputs"] else { return ["out": .object(preview)] }
        return outputs.mapValues { value in
            guard case .object(var tensor) = value else { return value }
            tensor["captureStatus"] = preview["captureStatus"]
            return .object(tensor)
        }
    }

    /// Follow captured boundary names to the captured primitive output. Configuration
    /// overrides change values, not boundary topology; unavailable ports stay absent.
    func resolvedPorts(instancePath: [String], localNodeID: String) -> [String: JSONValue]? {
        if let direct = ports(instancePath:instancePath,localNodeID:localNodeID) { return direct }
        guard let project, instancePath.count < 16 else { return nil }
        func definition(_ body:CompositeGraph,_ id:String) -> CompositeDefinition? {
            let instances = body.instances.filter { $0.id == id }
            guard instances.count == 1, let instance = instances.first else { return nil }
            let matches = project.definitions.filter { $0.reference == instance.definition }
            return matches.count == 1 ? matches.first : nil
        }
        var body = project.root
        for id in instancePath {
            guard let next = definition(body,id) else { return nil }
            body = next.body
        }
        func resolve(_ body:CompositeGraph,_ path:[String],_ node:String,_ port:String) -> JSONValue? {
            if let direct = ports(instancePath:path,localNodeID:node) { return direct[port] }
            guard path.count < 16, let child = definition(body,node) else { return nil }
            let outputs = child.outputs.filter { $0.id == port }
            guard outputs.count == 1, let source = outputs.first?.source else { return nil }
            return resolve(child.body,path+[node],source.componentID,source.portID)
        }
        guard let block = definition(body,localNodeID) else { return nil }
        var result: [String:JSONValue] = [:]
        for output in block.outputs {
            if let value = resolve(body,instancePath,localNodeID,output.id) { result[output.id] = value }
        }
        return result.isEmpty ? nil : result
    }

}
