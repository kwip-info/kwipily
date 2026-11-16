import Foundation

public struct GradientPolicyRecipe: Equatable {
    public let configuration: GradientPolicyConfiguration
    public let source: WorkflowNodeAddress
    public init(configuration: GradientPolicyConfiguration, source: WorkflowNodeAddress) {
        self.configuration = configuration; self.source = source
    }
}
public struct TrainabilityRecipe: Equatable {
    public let configuration: TrainabilityConfiguration
    public let source: WorkflowNodeAddress
    public init(configuration: TrainabilityConfiguration, source: WorkflowNodeAddress) {
        self.configuration = configuration; self.source = source
    }
}
/// Captured graph identity and declared tensor context only; native parameter membership
/// and differentiability are independently checked by the materialized worker model.
enum GradientWorkflowRecipeContext {
    static func validate(policy: GradientPolicyConfiguration?, trainability: TrainabilityConfiguration?, model: ModelExecutionSnapshot, registry: ComponentRegistry, context: GraphValidationContext) throws {
        guard policy != nil || trainability != nil else { return }
        let graph = model.graph
        let nodes = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        func selected(_ id: String) throws -> GraphNode {
            guard let node = nodes[id], model.origins[id] != nil else {
                throw workflowFailure(.configuration, "Selected gradient/trainability owner is not in this captured expanded model. Choose a current model node.")
            }
            return node
        }
        for raw in trainability?.owners ?? [] {
            guard case .object(let owner) = raw, let id = owner["node"]?.stringValue else {
                throw workflowFailure(.configuration, "Select an exact captured trainability owner.")
            }
            _ = try selected(id)
        }
        guard let policy, !policy.requests.isEmpty else { return }
        let validation = GraphValidator.validate(graph, registry: registry, context: context)
        for raw in policy.requests {
            guard case .object(let request) = raw, case .object(let output) = request["output"],
                  let outputNode = output["node"]?.stringValue, let outputPort = output["port"]?.stringValue,
                  case .array(let targets) = request["targets"] else {
                throw workflowFailure(.configuration, "Select the exact derivative output and targets.")
            }
            let node = try selected(outputNode)
            guard let definition = validation.configuredDefinition(nodeID:node.id,graph:graph,registry:registry),
                  definition.outputs.contains(outputPort), definition.configuredOutputs(node.config).contains(outputPort),
                  let contract = validation.outputTensorContracts[.init(node.id, outputPort)], contract.dtype == .float32 else {
                throw workflowFailure(.configuration, "Selected derivative output is not an available float32 port of this captured model.")
            }
            if request["reduction"] == .string("upstream") {
                guard case .object(let upstream) = request["upstream"], case .array(let shape) = upstream["shape"],
                      let rows = request["row_count"]?.integerValue,
                      shape == ([rows] + contract.layout.sample.dimensions).map({ .integer(Int64($0)) }) else {
                    throw workflowFailure(.configuration, "Upstream array must exactly match the selected output and observation count; broadcasting is not allowed.")
                }
            }
            for rawTarget in targets {
                guard case .object(let target) = rawTarget, let id = target["node"]?.stringValue else {
                    throw workflowFailure(.configuration, "Select an exact captured derivative target.")
                }
                let targetNode = try selected(id)
                if target["kind"] == .string("input") {
                    guard let port = target["port"]?.stringValue,
                          let definition = validation.configuredDefinition(nodeID:id,graph:graph,registry:registry),
                          definition.outputs.contains(port), definition.configuredOutputs(targetNode.config).contains(port) else {
                        throw workflowFailure(.configuration, "An input derivative target must be a genuine captured model input output port, not a detached intermediate value.")
                    }
                    if targetNode.type == "bag_input" {
                        let configuration = try BagSourceConfiguration(values: targetNode.config)
                        guard configuration.outputPorts.contains(port),
                              let coordinate = validation.outputBagContracts[.init(id, port)],
                              coordinate.sourceNodeID == id, coordinate.configuration == configuration,
                              coordinate.role.rawValue == port else {
                            throw workflowFailure(.configuration, "Select an enabled coordinate port of the captured bag input.")
                        }
                    } else {
                        guard ["input", "index_input"].contains(targetNode.type),
                              validation.outputTensorContracts[.init(id, port)] != nil else {
                            throw workflowFailure(.configuration, "An input derivative target must be a genuine captured model input output port, not a detached intermediate value.")
                        }
                    }
                }
                // Parameter names/shapes/aliases and frozen/integer/disconnected statuses are
                // native facts. Core deliberately does not invent them from field names.
            }
        }
    }
}
